#!/bin/bash
# Submit BioLiP (D_BIOLIP / B_BIOLIP) experiments: 5 models.
#
# Train pKd regression (ligand identities collapsed to one token; protein-only
# max readout; residue k-NN k=6 plus 8 A protein-ligand edges; batch 32
# molecules;
# 50 epochs; Cosmo/KPConv kernels [1.5, 4.5, 8.0], hidden 64; EGNN 256; SE3T 512;
# Cosmo max-pools frames onto nodes then protein), then (afterok)
# test affinity and |score| site attribution on the same checkpoint. After
# all tests, a CPU job writes the LaTeX table and Mol* attribution figure.
#
# Usage:
#   ./submit_biolip.sh train
#   ./submit_biolip.sh test
#   ./submit_biolip.sh plot

command=${1:-train}
if [ "${command}" != "train" ] && [ "${command}" != "test" ] && [ "${command}" != "plot" ]; then
    echo "Usage: $0 {train|test|plot}" >&2
    exit 1
fi

models=(gnn kpconv cosmo egnn se3_transformer)
partition=p.hpcl91
sbatch_gpu=(--gres=gpu:h100:1 --cpus-per-task 4 --ntasks-per-core 2 --mem-per-cpu 30G)
sbatch_cpu=(--cpus-per-task 12 --ntasks-per-core 2 --mem-per-cpu 4G)
plot_dir="results/_plots_biolip"
plot_wrap="python plot_biolip_table.py && python plot_biolip_attribution.py"

submit_job() {
    local out_dir=$1 wrap=$2 hours=$3
    shift 3
    local res=() extra=()
    if [ "$1" = "cpu" ]; then
        res=("${sbatch_cpu[@]}")
        shift
    else
        res=("${sbatch_gpu[@]}")
    fi
    extra=("$@")
    local job_id
    job_id=$(sbatch --parsable -p "${partition}" "${res[@]}" \
        --job-name=cosmo_biolip \
        --time "${hours}:00:00" \
        --output="${out_dir}/slurm-%j.out" \
        --error="${out_dir}/slurm-%j.out" \
        "${extra[@]}" \
        --wrap "${wrap}")
    echo "${job_id}  ${out_dir}" >&2
    echo "${job_id}"
}

submit_plot() {
    mkdir -p "${plot_dir}"
    local extra=()
    if [ "$#" -gt 0 ]; then
        extra+=(--dependency="afterok:$(IFS=:; echo "$*")")
    fi
    submit_job "${plot_dir}" "${plot_wrap}" 4 cpu "${extra[@]}"
}

if [ "${command}" = "plot" ]; then
    submit_plot
    exit 0
fi

test_ids=()
for seed in 1; do
    for model in "${models[@]}"; do
        exp_name="biolip_${model}_${seed}"
        out_dir="results/${exp_name}"
        mkdir -p "${out_dir}"
        args="configs/base.yaml configs/biolip.yaml configs/${model}.yaml seed=${seed} name=${exp_name}"
        test_wrap="bioverse test ${args} && python eval_biolip_attribution.py --name ${exp_name} --model ${model}"

        if [ "${command}" = "train" ]; then
            train_id=$(submit_job "${out_dir}" "bioverse train ${args}" 48)
            test_ids+=($(submit_job "${out_dir}" "${test_wrap}" 4 --dependency="afterok:${train_id}"))
        else
            test_ids+=($(submit_job "${out_dir}" "${test_wrap}" 4))
        fi
    done
done
submit_plot "${test_ids[@]}"
