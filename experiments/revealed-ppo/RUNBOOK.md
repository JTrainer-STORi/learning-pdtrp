# Training PDCTRPTW on revealed customers

The primary comparison is the difference in online greedy route cost between two policies trained from scratch. Each of three seeds initializes a pair of identical capacity-aware Transformers and generates identical instances for the two arms each epoch. The full arm encodes all customers during PPO. The revealed arm masks unannounced tokens in encoder attention and graph pooling, caches embeddings between calls, and records the revealed set independently of action feasibility for PPO replay.

Both arms use 50 customers, uniform training locations, DoD 0.2–0.8, a 10-hour call horizon, wide windows, 200 epochs of 12,800 instances, batch size 256, Adam at 0.0001, and four PPO passes. The objective is distance plus total service-start lateness. Both use the existing batched full-graph greedy **training baseline** for their advantage estimates. This holds the baseline procedure fixed while changing the information available to the actor. Recursive greedy validation selects the lowest equally weighted mean on fixed validation sets at DoD 0.2, 0.5 and 0.8, every two epochs and at the final epoch.

## One-time setup

Sync the updated checkout, including `experiments/revealed-ppo/` and `tests/`, to `$global_storage/learning-pdtrp` on the HEC.

From the **local checkout**, upload the prepared test data to your HEC home directory. Substitute your HEC username in the transfer command. [HEC login details](https://lancaster-hec.readthedocs.io/en/latest/login.html)

```bash
scp -r artifacts/revealed-ppo/data/26412 username@wayland-2022.hec.lancaster.ac.uk:
```

From **`$HOME` on the HEC**, install the launchers, stage the data, and verify the six test files. Each checksum should report `OK`.

```bash
cp "$global_storage/learning-pdtrp"/experiments/revealed-ppo/revealed-ppo-*.sh .
mkdir -p "$global_storage/learning-pdtrp/artifacts/revealed-ppo/data"
cp -R 26412 "$global_storage/learning-pdtrp/artifacts/revealed-ppo/data/"
(cd "$global_storage/learning-pdtrp/artifacts/revealed-ppo/data/26412" && sha256sum -c evaluation-inputs.sha256)
```

Training generates fresh instances each epoch. The uploaded directory is used only for evaluation. It contains 300 instances per combination of location family (uniform or ORTEC coordinates) and DoD (0.2, 0.5 or 0.8), for 1,800 instances in total. The directory name `26412` is the dataset seed.

## Submit from your HEC home directory

**1. Run the pilot.** This checks the PPO tests and trains the revealed policy for one epoch at seed 0.

```bash
sbatch revealed-ppo-pilot.sh
```

The pilot prints its training history to `~/revealed-pilot.JOB_ID.out`. Check `ppo_first_replay_max_abs_error`, `peak_gpu_allocated_bytes`, and `epoch_seconds`. Each production run has 24 hours for 200 epochs, an average budget of 432 seconds per epoch including startup and validation. Allow timing margin for either an L40 or an L40S. All three launchers request one GPU by count on `astro`. [HEC GPU queues and limits](https://lancaster-hec.readthedocs.io/en/latest/gpu.html)

**2. Train the six policies.** Submit after checking the pilot timing.

```bash
sbatch revealed-ppo-train.sh
```

Tasks 1–3 train full-graph seeds 0–2. Tasks 4–6 train revealed-customer seeds 0–2. Logs are in `~/revealed-train.ARRAY_ID_TASK.out`, with the array job number and task number filled in by Slurm. Checkpoints and histories are under `$global_storage/learning-pdtrp/artifacts/revealed-ppo/training/ARRAY_ID/`. The pilot has its own output directory.

**3. Evaluate after all six training tasks finish successfully.**

```bash
sbatch revealed-ppo-evaluate.sh
```

Evaluation uses the training array recorded at `training/production` and pins that array for all nine tasks. Tasks 1–6 use the six new best-validation checkpoints. Tasks 7–9 use the existing capacity-aware checkpoints from arrays 26338569 (seed 0) and 26342957 (seeds 1 and 2) under `artifacts/training-finetune/`. Each task verifies the uploaded files, then evaluates the same checkpoint and sample indices with greedy dispatch and with unweighted top-two rollout plus a legal depot return. Both modes see only revealed customers.

Evaluation logs are in `~/revealed-eval.ARRAY_ID_TASK.out`. Per-route results and decision timings are under `$global_storage/learning-pdtrp/artifacts/revealed-ppo/evaluation/ARRAY_ID/`.

## Reading the results

After evaluation completes, replace `EVAL_JOB_ID` with the evaluation array job number. The command writes both CSVs in that array’s results directory.

```bash
module add miniforge
"$global_storage/learning-pdtrp/.venv/bin/python" \
    "$global_storage/learning-pdtrp/experiments/revealed-ppo/summarize.py" \
    "$global_storage/learning-pdtrp/artifacts/revealed-ppo/evaluation/EVAL_JOB_ID"
```

If the archived reference checkpoints are unavailable but tasks 1–6 succeeded, append `--without-reference` to the summary command to report the from-scratch comparison.

`metrics.csv` reports mean objective and distance, mean total lateness, routes over 10 minutes of summed lateness, p95 objective, route wall time, and pooled p95 decision latency, by seed, method, width, and location/DoD. Rows with location `all` pool all six cells for one seed.

`paired.csv` gives per-seed changes in mean objective, relative changes, and paired wins/ties/losses (absolute tie tolerance 0.00001). Negative changes favor the candidate. Compare full versus revealed at width zero for the learned greedy policy, repeat at width two for the deployed method, and compare widths within each arm for the incremental value of search. The full/2 versus revealed/0 comparison measures how much of the control's search gain the new greedy actor recovers. Each training seed sees the same 1,800 test instances.
