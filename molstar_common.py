"""Shared Mol* viewer HTML, Chromium rendering, and cache helpers."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from plot_style import THEME_ACCENT

ROOT = Path(__file__).resolve().parent
MOLSTAR_VIEWER = ROOT / "node_modules" / "molstar" / "build" / "viewer"
PLAYWRIGHT_BROWSERS = ROOT / ".tools" / "pw-browsers"
PDB_DIR = ROOT / "figures" / "cache" / "pdb"

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RENDER_SIZE = 240
RENDER_CONCURRENCY = 4
CHROMIUM_ARGS = (
    "--allow-file-access-from-files",
    "--disable-dev-shm-usage",
    "--disable-gpu",
)
SURFACE_GREY = "#F0F0F0"


def download_pdb(pdb_id: str) -> Path:
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    path = PDB_DIR / f"{pdb_id.upper()}.cif"
    if path.exists():
        return path
    response = requests.get(RCSB_CIF_URL.format(pdb_id=pdb_id.upper()), timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _scores_digest(scores: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()


def _read_cache_meta(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_cache_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, sort_keys=True))


def _meta_is_current(meta: Optional[dict], checkpoint_meta: Optional[dict]) -> bool:
    if meta is None or checkpoint_meta is None:
        return False
    return (
        meta.get("checkpoint") == checkpoint_meta["checkpoint"]
        and meta.get("mtime") == checkpoint_meta["mtime"]
    )


def _hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _score_to_hex(score: float) -> str:
    t = float(np.clip(score, 0.0, 1.0))
    start = np.array([255.0, 255.0, 255.0])
    end = _hex_to_rgb(THEME_ACCENT)
    rgb = np.clip(start + t * (end - start), 0, 255).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def build_viewer_html(state_json: str, size: int = RENDER_SIZE) -> str:
    if not MOLSTAR_VIEWER.exists():
        raise FileNotFoundError(
            "Mol* viewer bundle not found. Run `npm install molstar` in the project root."
        )
    viewer_uri = MOLSTAR_VIEWER.resolve().as_uri()
    return f"""<!DOCTYPE html>
<html><head>
<script src="{viewer_uri}/molstar.js"></script>
<link rel="stylesheet" href="{viewer_uri}/molstar.css" />
<style>
html, body, #viewer1 {{
  margin: 0; padding: 0; width: {size}px; height: {size}px;
  overflow: hidden; background: #ffffff;
}}
.msp-plugin {{ background: #ffffff !important; }}
.msp-traj-controls,
.msp-state-snapshot-viewport-controls,
.msp-viewport-controls,
.msp-hover-box-wrapper,
.msp-logo {{
  visibility: hidden !important;
}}
</style></head><body><div id="viewer1"></div><script>
molstar.Viewer.create("viewer1", {{
  layoutIsExpanded: false,
  layoutShowControls: false,
  layoutShowRemoteState: false,
  layoutShowSequence: false,
  layoutShowLog: false,
  layoutShowLeftPanel: false,
  layoutShowRightPanel: false,
  viewportShowExpand: false,
  viewportShowSelectionMode: false,
  viewportShowAnimation: false,
  viewportShowControls: false,
  viewportShowReset: false,
  viewportShowSettings: false,
  viewportShowToggleFullscreen: false,
  viewportShowLogo: false,
  viewportShowTrajectoryControls: false,
  viewportShowScreenshotControls: false,
}}).then(viewer => {{
  window.__viewer = viewer;
  return viewer.loadMvsData({state_json}, "mvsj").then(() => {{
  const canvas3d = viewer.plugin.canvas3d;
  if (canvas3d) {{
    canvas3d.setProps({{ camera: {{ helper: {{ axes: {{ name: "off", params: {{}} }} }} }} }});
    canvas3d.requestDraw();
  }}
  window.__molstarDone = true;
  }});
}});
</script></body></html>"""


async def _render_one_panel(
    browser,
    html_path: Path,
    png_path: Path,
    size: int,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        page = await browser.new_page(
            viewport={"width": size, "height": size, "deviceScaleFactor": 2},
        )
        try:
            await page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=120_000)
            await page.wait_for_function("window.__molstarDone === true", timeout=120_000)
            await page.wait_for_timeout(500)
            data_url = await page.evaluate(
                """async () => {
  const canvas3d = window.__viewer?.plugin?.canvas3d;
  if (canvas3d) {
    canvas3d.setProps({ camera: { helper: { axes: { name: "off", params: {} } } } });
    canvas3d.requestDraw();
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }
  return document.querySelector("#viewer1 canvas").toDataURL("image/png");
}"""
            )
            png_path.write_bytes(base64.b64decode(data_url.split(",", 1)[1]))
        finally:
            await page.close()


def _render_concurrency(requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested))
    env = os.environ.get("MOLSTAR_RENDER_CONCURRENCY")
    if env:
        return max(1, int(env))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm))
    return max(1, min(RENDER_CONCURRENCY, os.cpu_count() or RENDER_CONCURRENCY))


async def _render_with_browser(playwright, jobs: list[tuple[Path, Path, int]]) -> None:
    if not jobs:
        return
    browser = await playwright.chromium.launch(headless=True, args=list(CHROMIUM_ARGS))
    semaphore = asyncio.Semaphore(1)
    try:
        await asyncio.gather(
            *[
                _render_one_panel(browser, html_path, png_path, size, semaphore)
                for html_path, png_path, size in jobs
            ]
        )
    finally:
        await browser.close()


async def _render_panels_batch(
    jobs: list[tuple[Path, Path, int]],
    *,
    concurrency: int | None = None,
) -> None:
    if not jobs:
        return
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(PLAYWRIGHT_BROWSERS)
    from playwright.async_api import async_playwright

    n_workers = min(_render_concurrency(concurrency), len(jobs))
    chunks = [jobs[i::n_workers] for i in range(n_workers)]
    async with async_playwright() as playwright:
        await asyncio.gather(
            *[_render_with_browser(playwright, chunk) for chunk in chunks if chunk]
        )
