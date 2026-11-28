# Training PDCTRPTW on revealed customers

The primary comparison is the difference in online greedy route cost between two policies trained from scratch. Each of three seeds initializes a pair of identical capacity-aware Transformers and generates identical instances for the two arms each epoch. The full arm encodes all customers during PPO. The revealed arm masks unannounced tokens in encoder attention and graph pooling, caches embeddings between calls, and records the revealed set independently of action feasibility for PPO replay.

Both arms use 50 customers, uniform training locations, DoD 0.2–0.8, a 10-hour call horizon, wide windows, 200 epochs of 12,800 instances, batch size 256, Adam at 0.0001, and four PPO passes. The objective is distance plus total service-start lateness. Both use the existing batched full-graph greedy **training baseline** for their advantage estimates. This holds the baseline procedure fixed while changing the information available to the actor. Recursive greedy validation selects the lowest equally weighted mean on fixed validation sets at DoD 0.2, 0.5 and 0.8, every two epochs and at the final epoch.

## Current actor as the greedy PPO baseline

After syncing the updated checkout, run these commands from your HEC home directory. The job reuses the six uploaded datasets in `data/26412` and the raw-score checkpoints in `raw/26479898`, both under `$global_storage/learning-pdtrp/artifacts/revealed-ppo/`.

```bash
cp "$global_storage/learning-pdtrp/experiments/revealed-ppo/revealed-ppo-baseline.sh" .
sbatch revealed-ppo-baseline.sh
```

Each array task handles one seed (0, 1 or 2) on one GPU on `astro`. The actor uses revealed-customer encoding and raw decoder scores in every arm. Only the way PPO obtains the greedy route cost for its advantage changes.

| Training baseline | Weights used for the greedy route | Customers visible to its encoder |
|---|---|---|
| `rollout` (archived control) | Snapshot promoted after a paired test on separate baseline instances | Full graph |
| `current-full` | Actor that sampled this PPO collection group | Full graph |
| `current-revealed` | Actor that sampled this PPO collection group | Revealed customers |

For both new arms, PPO samples a route and greedily decodes the same training instance, including the same realized service times. It computes `greedy cost − sampled cost` once before the first optimizer step for the group. All four PPO passes reuse that advantage. A sampled route gets positive advantage when it improves on the greedy route. The revealed greedy baseline updates its encoder inputs as calls arrive. The full baseline can use unannounced customers in its embeddings, although its dispatcher can only select available actions.

Both new arms train from scratch for 40 epochs of 12,800 routes per seed (512,000 total per run). They share model initialization seeds, per-epoch training instances, optimizer settings and the original validation rule. Recursive greedy validation every two epochs selects the checkpoint. The archived control ran for 200 epochs, but its winning checkpoints occurred at zero-based epochs 15, 15 and 3. These are also the checkpoints that won within the first 40 epochs. The launcher verifies the archived digest and selection window, then evaluates the control and both new checkpoints afresh on all 1,800 routes using greedy dispatch, zero search candidates and one evaluator process.

Logs appear in `~/revealed-baseline.ARRAY_ID_SEED.out`. Results appear in `$global_storage/learning-pdtrp/artifacts/revealed-ppo/baseline/ARRAY_ID/seed-SEED/`. Each seed writes the following results.

- `paired.csv` reports candidate minus control cost for each location and DoD, and across all 1,800 routes. Negative values favor the candidate. Compare `current-full` against `rollout` for the effect of refreshing the weights, then `current-revealed` against `current-full` for the effect of hiding future customers. The third comparison gives the combined effect.
- `metrics.csv` includes mean and p95 cost, distance, lateness and decision latency. Each new model's `history.csv` records sampled cost, baseline cost, the fraction of routes with positive advantage, sampled relative entropy and validation costs. A falling fraction of positive advantages together with near-zero entropy can indicate that sampled and greedy routes have converged. Inspect validation and test cost to judge whether that convergence helped.
- Use the three per-seed overall cost changes as the training replicates. If the revealed baseline improves on the control consistently, compare its absolute greedy cost with the prior clipped revealed PPO result (54.27) before extending training. The clipped result sets the performance threshold for this decoder experiment.

## Raw decoder scores for greedy revealed PPO

After syncing this checkout to the HEC, run these two commands from your HEC home directory.

```bash
cp "$global_storage/learning-pdtrp/experiments/revealed-ppo/revealed-ppo-raw.sh" .
sbatch revealed-ppo-raw.sh
```

Array tasks 0–2 each train one revealed PPO policy from scratch with `--tanh_clipping 0`. The decoder feeds its raw scores to the masked softmax. This tests removal of the entire `10 × tanh` transform, including its factor of ten near zero. PPO still samples actions during training and uses the same clipped probability-ratio objective as before. Each seed gets the same initialization procedure, epoch data seeds, 200 × 12,800 instances, four update passes, and recursive greedy validation rule as its clipped counterpart in training array `26442619`.

Each task then evaluates its best-validation checkpoint with greedy dispatch on all 1,800 instances in the existing `data/26412` directory. It compares these routes with the same seed and sample IDs from the clipped revealed policies in `evaluation/26469992`. Both directories are under `$global_storage/learning-pdtrp/artifacts/revealed-ppo/`. The script verifies the test set against the archived checksums before training. The three tasks each request one GPU by count on `astro`. [HEC GPU guide](https://lancaster-hec.readthedocs.io/en/latest/gpu.html)

Logs are in `~/revealed-raw.ARRAY_ID_SEED.out`. Results are in `$global_storage/learning-pdtrp/artifacts/revealed-ppo/raw/ARRAY_ID/seed-SEED/`. Each log ends with the paired mean greedy costs. Each seed directory contains the following files.

- `paired.csv` gives raw minus clipped cost, percentage change, and wins/ties/losses for each location/DoD and for all 1,800 routes. Negative change favors raw scores. Treat the three seeds as three training replicates.
- `metrics.csv` gives mean and p95 cost, distance, mean lateness, number of routes over ten minutes late, and route and decision timing for both decoders. GPU models are recorded in the evaluation manifests.
- `models/*/*/history.csv` records validation cost, PPO approximate KL, clip fraction and replay error through the final epoch. Inspect these alongside the selected epoch in `best-validation.json` to see whether late training remained stable. Each epoch also records `ppo_sampled_relative_entropy`, the mean sampled negative log probability divided by the log of the number of legal actions at that decision. It uses the first collection batch and counts only active decisions with a choice (`ppo_entropy_decisions`). Values near one suggest roughly uniform sampling, and values near zero suggest nearly deterministic choices. Sampling noise can take the estimate above one. `candidates-0/` contains the per-route results. The evaluation manifest now records the loaded decoder's `tanh_clipping` value.

On this benchmark the prior three-seed mean greedy costs were 54.27 for clipped revealed PPO, 18.05 for full-graph PPO and 17.25 for the capacity-aware reference. The comparison reuses a previously inspected benchmark so the route-level differences are directly comparable. Confirm any chosen configuration on a fresh dataset seed before treating the gain as a final performance estimate.

## Original clipped experiment — one-time setup

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
