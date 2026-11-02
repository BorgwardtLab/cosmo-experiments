#!/bin/bash
# Submit toy (Picasso ship) experiments: 5 models x 5 perturbations.
#
# After all tests, a GPU job writes example / performance / attribution figures.
#
# Usage:
#   ./submit_toy.sh train
#   ./submit_toy.sh test
#   ./submit_toy.sh plot

command=${1:-train}
if [ "${command}" != "train" ] && [ "${command}" != "test" ] && [ "${command}" != "plot" ]; then
    echo "Usage: $0 {train|test|plot}" >&2
    exit 1
fi

if [ -n "${TOY_MODELS:-}" ]; then
    read -r -a models <<< "${TOY_MODELS}"
else
    models=(gnn kpconv cosmo egnn se3_transformer)
fi
if [ -n "${TOY_SHIPS:-}" ]; then
    read -r -a ships <<< "${TOY_SHIPS}"
else
    ships=(ship00 shipr0 shipch shippc shiptl)
fi
cfgs="configs/base.yaml configs/ship.yaml"
partition=p.hpcl91
sbatch_gpu=(--gres=gpu:h100:1 --cpus-per-task 2 --ntasks-per-core 2 --mem-per-cpu 10G)
sbatch_cpu=(--cpus-per-task 4 --ntasks-per-core 2 --mem-per-cpu 8G)
plot_dir="results/_plots_toy"
plot_wrap="PYTHONUNBUFFERED=1 python -u plot_toy_example.py && python -u plot_toy_performance.py && python -u plot_toy_attribution.py && python -u plot_toy_attribution_mass.py"

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
        --job-name=cosmo_toy \
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
    submit_job "${plot_dir}" "${plot_wrap}" 2 "${extra[@]}"
}

if [ "${command}" = "plot" ]; then
    submit_plot
    exit 0
fi

test_ids=()
for seed in 1; do
    for model in "${models[@]}"; do
        for ship in "${ships[@]}"; do
            exp_name="toy_${model}_${ship}_${seed}"
            out_dir="results/${exp_name}"
            mkdir -p "${out_dir}"
            args="configs/base.yaml configs/${model}.yaml configs/ship.yaml configs/${ship}.yaml seed=${seed} name=${exp_name}"
            if [ "${model}" = "cosmo" ] || [ "${model}" = "kpconv" ]; then
                args="${args} hidden_dim=64"
            fi

            if [ "${command}" = "train" ]; then
                train_id=$(submit_job "${out_dir}" "bioverse train ${args}" 24)
                test_ids+=($(submit_job "${out_dir}" "bioverse test ${args}" 4 --dependency="afterok:${train_id}"))
            else
                test_ids+=($(submit_job "${out_dir}" "bioverse test ${args}" 4))
            fi
        done
    done
done
submit_plot "${test_ids[@]}"
