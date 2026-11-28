# Score one checkpoint on the six fixed test cells. Call with checkpoint, data and output paths.
evaluate_greedy() {
    local checkpoint=$1 data=$2 output=$3 location dod
    local -a datasets
    for location in uniform-uniform ortec-locations; do
        datasets=()
        for dod in 0.2 0.5 0.8; do
            datasets+=("$data/$location/pdctrptw50-dod$dod-th10-seed26412.txt")
        done
        OMP_NUM_THREADS=2 srun --ntasks=1 --cpus-per-task=2 --export=ALL --cpu-bind=cores \
            python -u evaluate_trained_policy.py --problem pdctrptw --checkpoint "$checkpoint" \
            --datasets "${datasets[@]}" --num-samples 300 --speed 4 --time-horizon 10 --latest-end 2 \
            --evaluation-seed 12346 --device cuda --record-decision-times --rollout-candidates 0 \
            --output-dir "$output/candidates-0/$location"
    done
}
