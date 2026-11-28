# Score one checkpoint on the six fixed test cells.
evaluate_greedy() {
    local checkpoint=$1 data=$2 output=$3 dataset_seed=${4:-26412} location dod
    local -a datasets
    for location in uniform-uniform ortec-locations; do
        datasets=()
        for dod in 0.2 0.5 0.8; do
            datasets+=("$data/$location/pdctrptw50-dod$dod-th10-seed$dataset_seed.txt")
        done
        OMP_NUM_THREADS=2 srun --ntasks=1 --cpus-per-task=2 --export=ALL --cpu-bind=cores \
            python -u evaluate_trained_policy.py --problem pdctrptw --checkpoint "$checkpoint" \
            --datasets "${datasets[@]}" --num-samples 300 --speed 4 --time-horizon 10 --latest-end 2 \
            --evaluation-seed 12346 --device cuda --record-decision-times --rollout-candidates 0 \
            --output-dir "$output/candidates-0/$location"
    done
}

evaluate_greedy_case() {
    local checkpoint=$1 data=$2 output=$3 dataset_seed=$4
    mkdir -p "$output"
    sha256sum "$checkpoint" > "$output/checkpoint.sha256"
    cp "$data/evaluation-inputs.sha256" "$output/"
    cp "${checkpoint%.pth}.json" "$output/selection.json"
    evaluate_greedy "$checkpoint" "$data" "$output" "$dataset_seed"
}
