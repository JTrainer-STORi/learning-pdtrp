# Training PDCTRPTW on revealed customers

The primary comparison is the difference in online greedy route cost between two policies trained from scratch. Each of three seeds initializes a pair of identical capacity-aware Transformers and generates identical instances for the two arms each epoch. The full arm encodes all customers during PPO. The revealed arm masks unannounced tokens in encoder attention and graph pooling, caches embeddings between calls, and records the revealed set independently of action feasibility for PPO replay.

Both arms use 50 customers, uniform training locations, DoD 0.2–0.8, a 10-hour call horizon, wide windows, 200 epochs of 12,800 instances, batch size 256, Adam at 0.0001, and four PPO passes. The objective is distance plus total service-start lateness. Both use the existing batched full-graph greedy **training baseline** for their advantage estimates. This holds the baseline procedure fixed while changing the information available to the actor. Recursive greedy validation selects the lowest equally weighted mean on fixed validation sets at DoD 0.2, 0.5 and 0.8, every two epochs and at the final epoch.

## HEC submission

Sync the repository changes, including this directory and the tests, to the checkout containing the activated project environment on HEC. From your HEC home directory set:

```bash
repo="$global_storage/learning-pdtrp"
experiment="$repo/experiments/revealed-ppo"
results="$repo/artifacts/revealed-ppo"
```

Run one production-sized epoch of the revealed arm and generate the independent test data:

```bash
pilot=$(sbatch --parsable --array=4 --time=02:00:00 "$experiment/train.sh" pilot)
data=$(sbatch --parsable "$experiment/data.sh")
echo "Pilot $pilot, data $data"
```

The default ORTEC source is `new_data/ortec_rl/ORTEC-VRPTW-ASYM-00c5356f-d1-n258-k12.txt` under the repository. Supply an absolute source path as the first argument to `data.sh` if stored elsewhere. The job generates 300 instances in each of six location/DoD cells with seed 26412, records their hashes, and publishes them under `data/DATA_JOB_ID`. ORTEC contributes coordinates. Calls, demands, service durations, and windows use the synthetic benchmark rules.

The pilot runs the replay and visibility regression on CUDA and writes a normal training history. Inspect it before choosing the production time request:

```bash
cat "$results/training/$pilot/revealed/seed-0"/models/*/*/history.csv
```

The pilot, training and evaluation each request one GPU on `astro`. Slurm can allocate either an L40 or an L40S. Check the recorded GPU model when comparing pilot and production timings. Inspect `ppo_first_replay_max_abs_error`, `peak_gpu_allocated_bytes`, and `epoch_seconds` in the pilot history. The default 24-hour training request gives each 200-epoch run an average of 7 minutes 12 seconds per epoch, before job startup. Use the pilot timing to estimate the total, allowing for variation in validation and baseline updates. If the run fits comfortably, you can shorten the production request with `sbatch --time`. If the estimate exceeds 24 hours, choose a longer queue and a GPU type available there when submitting. [Lancaster HEC GPU queues and model requests](https://lancaster-hec.readthedocs.io/en/latest/gpu.html)

Submit the six fresh training tasks and schedule the nine evaluation tasks:

```bash
training=$(sbatch --parsable "$experiment/train.sh")
evaluation=$(sbatch --parsable --dependency="afterok:$training:$data" \
    "$experiment/evaluate.sh" "$training" "$data")
echo "Training $training, evaluation $evaluation"
```

Training tasks 1–3 are full-graph seeds 0–2 and tasks 4–6 are revealed seeds 0–2. Evaluation uses the same mapping and adds reference tasks 7–9 for the earlier capacity-aware checkpoints from arrays 26338569 and 26342957. Every task evaluates one frozen checkpoint twice on the same dataset bytes and sample indices, first with greedy dispatch and then with unweighted top-two customer rollout plus the legal depot return. All inference modes see revealed customers only. If the archived checkpoints are unavailable, use `--array=1-6` on the **evaluation** submission and `--without-reference` when summarizing.

## Reading the results

After evaluation completes:

```bash
"$repo/.venv/bin/python" "$experiment/summarize.py" "$results/evaluation/$evaluation"
```

`metrics.csv` reports mean objective and distance, mean total lateness, routes over 10 minutes of summed lateness, p95 objective, route wall time, and pooled p95 decision latency, by seed, method, width, and location/DoD. Rows with location `all` pool all six cells for one seed.

`paired.csv` gives per-seed changes in mean objective, relative changes, and paired wins/ties/losses (absolute tie tolerance 0.00001). Negative changes favor the candidate. Compare full versus revealed at width zero for the learned greedy policy, repeat at width two for the deployed method, and compare widths within each arm for the incremental value of search. The full/2 versus revealed/0 comparison measures how much of the control's search gain the new greedy actor recovers. Each training seed sees the same 1,800 test instances.
