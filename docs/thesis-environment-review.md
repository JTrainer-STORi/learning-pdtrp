# PDTRP environment review against Larsen's thesis

## Conclusion and scope

The repository implements a recognizable adaptation of the problems in Allan
Larsen's *The Dynamic Vehicle Routing Problem* (2000), rather than an exact
reproduction. The distance and soft-lateness objectives largely match chapters
5 and 7. The service-time information leak identified in finding 2 has been
corrected: online OR-Tools uses expected service durations, and MSA samples
scenario durations. Dataset generation differs from the thesis, and masked-mode
training deliberately uses a different information set to control training
memory. Departure commitment has now been explicitly
adopted: new calls during pre-departure waiting allow reconsideration, while
travel and service remain non-interruptible.

This review covers the four canonical classes in `problems/`, their generators,
policy observations, and the online/MSA OR-Tools planning interfaces. A follow-up
also checks §§4.2 and 6.2.2 specifically for the meaning of reaction time; this
does not constitute a full chapter 6 conformance review. Chapter 7
defines **DTSPTW**, not the repository's capacitated PDCTRPTW. Capacity/reloading
are extensions; their fidelity cannot be certified from chapters 5 and 7.
This is a code-and-counterexample review, not a reproduction of the thesis's
experimental tables or a proof that every execution path is correct.

The original review did not change runtime behavior. The follow-up correction
for finding 2 changes planning inputs; the simulator and scoring continue to use
realized service durations. The follow-up change for finding 1 introduces
departure commitment in both TW environments. Other findings retain their
separate status. Initial empty queues are now handled in all four variants.
Recursive model evaluation explicitly requires one instance and one rollout
per call; dataset evaluation processes instances sequentially.

## Source and page references

The source is the supplied [imm143.pdf](../imm143.pdf). Page references below
are the thesis's **printed page numbers**. In this PDF, the corresponding viewer
page number is 17 larger:

| Topic | Thesis section and printed pages | PDF viewer pages |
| --- | --- | --- |
| PDTRP definition and information assumptions | 5.1, 66–67 | 83–84 |
| PDTRP generation and service distributions | 5.2, 68–70 | 85–87 |
| DTSPTW definition and cost | 7.1, 108–111 | 125–128 |
| Event simulation and expected service times | 7.2, 111–114 | 128–131 |
| Idle/repositioning policies | 7.3, 114–116 | 131–133 |
| Generation and experimental parameters | 7.4, 116–119 | 133–136 |
| Reaction-time definition | 4.2, 59 | 76 |
| Minimum 60-minute opening delay, Scenario C | 6.2.2, 91 | 108 |

## Confirmed matches

| Thesis requirement | Implementation and qualification |
| --- | --- |
| Minimize routing cost for PDTRP, rather than classical repairman latency; §5.1, p. 66 | `PDTRP.get_costs` minimizes distance. The repository additionally makes the tour explicitly closed by charging the final return. |
| Advance requests are known initially; immediate requests are revealed later; §§5.1 and 7.1 | State initialization distinguishes zero and positive arrival times; action masks exclude unrevealed customers. An initially empty queue waits for its first call. |
| Constant travel speed and no diversion during travel; §5.1 | Updates use distance divided by speed. Pre-departure waits can be interrupted by calls; travel and service remain non-interruptible. |
| No service before a time window opens; §7.1.1, p. 109 | TW state updates use `max(current_time + travel_time, window_start)` as service start. |
| Soft upper time windows with linear lateness penalties; §7.1.3, equations 7.1–7.2 | TW scoring uses `theta * closed_tour_distance + gamma * sum(max(0, service_start - window_end))`. Late service remains possible. Capacitated scoring excludes depot visits from customer lateness. |
| Service times log-normal with mean 3 minutes and variance 5; §§5.2 and 7.4 | Generated service times use `sample_lognorm(3, sqrt(5), ...)`; its parameter conversion is correct, with an additional rounding step. |
| Every request must be served; p. 118 | Tour validators require every customer, and completion depends on visiting all customers. |
| Calls stop at T, but service can continue afterwards; Algorithm 3, p. 112 | `time_horizon` does not impose a hard route-completion deadline. Soft windows can also be exceeded with a penalty. |

Relevant implementations: [PDTRP scoring](../problems/pdtrp/problem_pdtrp.py#L21),
[TW scoring](../problems/pdtrp_tw/problem_pdtrptw.py#L20),
[capacitated TW scoring](../problems/pdctrp_tw/problem_pdctrptw.py#L21), and
[log-normal sampling](../problems/utils.py#L43).

## Findings that affect faithful online simulation

### 1. Resolved design choice: commitment now binds at departure

**Update:** Requests arriving during a pre-departure wait now trigger a decision
at the vehicle's current location. A call exactly at departure is revealed before
committing. Once departure occurs, travel and service are non-interruptible.
The example below documents the old behavior; the updated diagnostic reaches
the next decision at t=10, executes `[0, 2, 1]`, and incurs zero lateness in both
TW variants. Online scores preserve the actual time spent waiting, and canceled
choices are excluded from tours and capacity usage.

**Thesis:** Service cannot begin early, and the vehicle waits at its preceding
location before departure (p. 109). Algorithm 3 handles new-customer events
separately and re-solves the current route (pp. 111–112). This suggests that a
request arriving during a pre-departure wait could change the next customer.
However, the pseudocode does not fully specify commitment and event precedence.
The explicit chapter 5 restriction forbids changing destination **while
traveling** (p. 66); it does not explicitly extend that restriction to the whole
waiting interval.

**Code before the change:** A single TW `update(selected)` advanced through
travel, any window wait, and service. Arrivals in that entire interval were
only reported at the end. This implemented commitment **at selection**,
including any wait before service, and was correct under that former rule. See
[StatePDTRPTW.update](../problems/pdtrp_tw/state_pdtrptw.py#L99),
[StatePDCTRPTW.update](../problems/pdctrp_tw/state_pdctrptw.py#L109), and the shared
[departure-event implementation](../problems/time_window_events.py#L28).

**Effect of the two rules:** At t=0, only customer A is known. A is one minute away but
its window opens at t=100. B calls at t=10, is 0.1 minutes from the depot, and
has window [10, 12]. Selecting A makes the next state decision occur at t=100.
B is then served at t=100.9, incurring 88.9 minutes of lateness. An event-based
dispatcher waiting at the depot could instead reconsider at t=10, serve B at
t=10.1, and still serve A at t=100, with zero lateness and the same total distance.
Both TW variants reproduce this behavior. The alternative schedule assumes
commitment can be revised before departure; it is not a valid alternative under
the former commitment-at-selection rule.

The arithmetic service-start formula is correct for an already fixed route.
The earlier classification of this example as a definite bug was too strong.
The adopted rule is now commitment at departure, with explicit arrival events
during waiting. Neither interpretation requires diversion during travel or
interruption of ongoing service.

### 2. Resolved: unserved customers' realized service times reached online planning

**Thesis:** The realized service duration is unknown until service completes
(pp. 66, 108–109). The chapter 7 planner uses the **expected** duration when
evaluating candidate schedules (pp. 113–114).

**Original issue:** The online scenario builder copied actual service durations
for unserved customers into its solve. MSA did the same for already revealed,
unserved customers. This exposed information unavailable under the thesis's
definition; pre-sampling durations for the simulator itself was valid.

**Current behavior:** [online_scenario](../evaluate_online_ortools.py) uses the
known distribution mean for unserved customer durations.
[make_scenario](../evaluate_msa_ortools.py) independently samples log-normal
durations for every unserved customer in each scenario, including known
customers and future customers drawn from an ORTEC pool. Sampling happens in
the parent process before parallel solves. Distribution parameters are prior
metadata, with defaults of mean 3 minutes and variance 5 minutes²; evaluation
accepts `--service-times-mean` and `--service-times-var` to configure that prior.
Neither approach estimates it from the realized unserved durations.

The updated probe supplies an actual duration of 12 minutes and a prior mean
of 3. It reports 3 for online planning and 12 for the deliberately clairvoyant
scenario. Realized durations still advance the simulation and determine scored
costs. The neural encoder does not directly embed service durations in
`_init_embed`, so the original issue should not be described as identical service
time leakage in every method. The clairvoyant reference retains realized
durations as an explicitly different information set.

### 3. Intentional training approximation: one GNN encoding to limit memory use

**Thesis:** Actual immediate-request locations become known when calls arrive;
knowing regional arrival intensities in advance (§7.3) does not reveal individual
future requests.

**Code:** [forward's masked branch](../nets/dynamic_attention_model.py#L245)
encodes the complete input graph, including unrevealed nodes, before masking
actions. [Training](../train.py#L342) invokes the default `mode='masked'`.
Thus unavailable customers cannot normally be selected, but can affect the
embeddings and probabilities of available customers. The constructor documents
this design.

**Maintainer clarification:** This was deliberately chosen to avoid excessive
GNN memory use during gradient-based training. One encoder pass is reused across
the rollout. Re-encoding after each arrival in a differentiable training rollout
can retain several encoder computation graphs and their saved activations until
the trajectory loss is backpropagated. The saving concerns those activations and
autograd graphs, rather than repeated parameter-gradient buffers. Decoder
rollout memory still grows with the trajectory.

This is an intentional training approximation, not an unintended environment
implementation defect. It should be documented as a memory/information tradeoff.
The current recursive branch is designed for evaluation; switching training to
it is not a drop-in replacement for the masked training path.

**Illustration of the information difference:** With weights seeded at 1234,
changing only one unrevealed
customer's location changes the first known customer's selection probability
from approximately 0.2687 to 0.1904. The corresponding `recursive` calculation
is unchanged (approximately 0.3194 in both inputs).

The current [trained-policy evaluator](../evaluate_trained_policy.py#L91)
uses `recursive`, and
[_mask_input](../nets/dynamic_attention_model.py#L551) rebuilds the graph over
revealed nodes. This finding does **not** establish that those evaluation
results leak future locations. The training and evaluation information sets
differ by design. Report online policy performance using the recursive path and
describe the masked training procedure explicitly; masked-mode runs themselves
should not be presented as strictly online results.

### 4. Resolved: all four variants support an initially empty queue

**Thesis:** The chapter 5 and 7 experiments include 100% dynamism (pp. 70, 119).
There can be no advance requests and an initially empty queue.

**Original issue:** PDTRP and PDTRPTW started at t=0 with every action masked
when every customer called later. Idle fast-forward occurred only after an
action had already been selected. The capacitated variants had a depot action
and did not have the identical failure.

**Current behavior:** All four initializers now use
[reveal_and_wait_if_idle](../problems/request_events.py). Each empty batch row
advances to its own first call, reveals simultaneous calls together, and stays
at the depot without travel, capacity consumption, or an extra recorded visit.
Rows with advance requests stay at t=0. The diagnostic now finds a legal action
for all four variants with calls at t=10 and t=20.

[Fully dynamic regressions](../tests/test_fully_dynamic.py) cover generation,
mixed-batch initialization, model-free policies, neural rollouts, POMO starts,
and online/MSA evaluation. POMO cycles through available starting customers
when fewer customers have been revealed than the number of rollouts.

## Dataset and modeling differences

### 5. Intentional adaptation: bounded training sizes

**Design clarification:** Fixed request counts were deliberately chosen to
bound training instance sizes and simplify batching. This is an intentional
benchmark design choice, not an implementation mistake. The Poisson
distribution has unbounded support; fixed training sizes control memory costs
for graph-based models.

The thesis generates a stochastic number of immediate requests from Poisson
processes (§5.2.1, p. 69; §7.1.2, p. 110). The repository first chooses a total
size and a bounded integer number of dynamic customers. The shared
[location generator](../problems/location_generation.py#L108) accepts Poisson
regional counts only when their sum equals that preselected total, and then
generates request times using a scaled beta distribution.

Uniform times with a fixed count are compatible with a Poisson process
**conditioned on that count**. Therefore uniform draws alone are not evidence
of an incorrect implementation. However, fixed total/dynamic counts, the
one-minute lower arrival bound, optional beta-skewed times, and optional random
Dirichlet spatial weights define a different benchmark distribution from the
thesis's unconditioned generation. Nonuniform temporal skews are additional
extensions, not homogeneous Poisson arrivals.

These are reasonable experimental choices if documented. Use one subregion or
explicit uniform weights for the chapter 5 spatial setting; the default random
regional weights do not reproduce its uniform service region. Public N now
counts customers excluding the depot, matching the thesis. The new
`min_customers` / `max_customers` dataset arguments and customer-count CLIs
follow that convention; explicitly named legacy total-node bounds retain their
old meanings. Earlier saved instances are not resized or relabelled.
Integer rounding affects the achieved degree of dynamism.

**Proposed evaluation extension:** Keep fixed-size training and existing
fixed-size test sets, and add separate Poisson-arrival evaluation sets. For a
fixed advance count `n_adv`, draw `n_dyn ~ Poisson(lambda * T)`, then draw uniform
call times on `[0, T]` and sort them. This constructs the homogeneous process
over a finite horizon. Choose `lambda = (mu - n_adv) / T` for expected total
customer count `mu`, consistent with equation 5.1. Spatial subregions can use
separate rates. Record realized counts and dynamism; an exact dynamic fraction
will generally vary between instances.

Current policy and OR-Tools evaluators process instances individually, which
supports evaluation files with varying row sizes. A generator still needs to
handle small counts, capacity-demand generation, and any resource limits.
If oversized instances are capped or rejected, label the resulting distribution
as truncated/conditioned and report the exclusions. Save generated instances
once and use identical files for every comparison method. Comparing fixed-size
and Poisson evaluation separately distinguishes performance on the training
distribution from generalization to stochastic workload sizes.

### 6. Resolved: synthetic windows now use chapter 7 openings and width presets

**Implementation update:** Both synthetic TW generators now use
`e_i = t_i + distance(depot, i) / speed` for immediate requests. Advance-request
openings remain random and no earlier than direct depot arrival. The shared
[time-window generator](../problems/time_windows.py) supports `narrow` (30–90
minutes) and `wide` (60–180 minutes, default); explicit bounds override presets.
MSA receives the configured bounds as prior metadata, and its CLI exposes the
same presets. Fixed request counts remain intentional.

Closing times are drawn uniformly from `[e_i + minimum,
min(e_i + maximum, T + latest_end)]`. This is an explicit implementation
convention for the underspecified closing-time sampling, rather than a claim to
reproduce the thesis's exact random generator or its wider average immediate
windows. Configurations that cannot fit the minimum width raise an error.
The legacy reaction-time setting no longer controls synthetic TW openings.
Existing dataset files and ORTEC source windows are retained as supplied.

The following explains the motivation for the previous rule and the distinction
between thesis scenarios.

**Correction to the original review:** A 60-minute delay does have an explicit
precedent in this thesis. In §6.2.2, p. 91 (PDF page 108), Scenario C opens
immediate-request windows at call time plus a random delay of at least 60
minutes, described there as a minimum reaction time. The repository's fixed
60-minute opening delay is a related modeling choice. It is not an unsupported
addition, although a fixed delay does not reproduce that random-delay scheme
exactly.

The thesis also uses a formal urgency measure in §4.2, p. 59:
`r_i = l_i - t_i`, the time from the call to the **latest** service start.
The legacy repository parameter instead controls a minimum delay to the **earliest**
service start. These quantities differ: a 60-minute opening delay followed by
an 80-minute window gives a formal reaction time of 140 minutes, provided the
depot-travel lower bound does not increase the opening delay.

Chapter 7 uses another scenario construction: on p. 117, immediate-request
windows open at call time plus travel time from the depot:
`e_i = t_i + travel(depot, i)`.

The previous [synthetic TW generator](../problems/pdtrp_tw/problem_pdtrptw.py)
used `max(travel(depot, i), t_i + reaction_time)`, with a default
60-minute reaction time. The
[capacitated generator](../problems/pdctrp_tw/problem_pdctrptw.py#L305)
used the same construction. In the original seeded probe, all nine dynamic windows opened
60 minutes after the call; depot travel times range from about 1.62 to 8.42
minutes. These are different urgency settings; the comparison establishes a
difference from chapter 7's generator, not an error in adopting a
chapter-6-inspired opening delay. Document the parameter as a minimum
notification-to-opening delay, while retaining its existing CLI name for
compatibility unless a separate rename is requested.

The former default widths 60–100 differed from thesis scenarios 30–90 and
60–180 (Table 7.1); those two presets are now available. ORTEC-derived instances use a separate generation path;
this particular synthetic-window finding is not a claim about every saved file.

### 7. Extensions and conventions requiring explicit documentation

- **Capacity:** PDCTRP/PDCTRPTW add demand limits and depot reloads. Chapters 5
  and 7 provide no specification for validating those additions. Chapter 6 would
  require a separate review if its capacitated model is intended as a source.
- **Idle positioning:** The uncapacitated environment supports staying in place
  when its queue empties, but no selectable idle-point/repositioning system.
  This resembles chapter 7's NO REPOS baseline, with the waiting-event caveat,
  rather than its NEAREST/BUSIEST/HI-REQ variants.
- **Time capacity:** The TW classes impose soft deadlines, not a hard maximum
  route duration. `time_horizon` is chiefly the arrival-generation horizon.
- **Geometry and units:** A unit square with speed 4 units/hour is consistent
  with the intended 10-km square and 40-km/hour speed after normalization.
  Distances and cost weights still need consistent units before comparing
  numerical objectives with thesis tables. Do not copy literal weights across
  different distance units without conversion.
- **End conditions:** Repository scoring includes return-to-depot distance and
  ends an episode once all stored customers are visited. §5.1 does not explicitly
  settle that final-return convention; Algorithm 3 additionally waits until the
  calling horizon has ended. These details need an agreed benchmark definition.
- **Thesis ambiguities:** Chapter 5 p. 68 describes a 10-km region but also gives
  a conflicting scaling statement; chapter 7 p. 116 gives the consistent
  1000-units/km convention. Chapter 7's description of adjusting window starts
  is clearer than its exact distributional construction of closing times. Those
  details should not be silently resolved and labeled verified.

## Recursive evaluation scope and the remaining state-slicing limitation

[DynamicAttentionModel.forward](../nets/dynamic_attention_model.py#L209)
rejects `recursive` and `recursive_plus_removal` calls unless the input batch
contains exactly one instance and `pomo_batch_size=1`. Invalid calls raise a
`ValueError` before encoding or state construction. Masked training continues
to support full batches and multiple POMO rollouts.

The policy evaluator already uses a `DataLoader` with `batch_size=1`.
`--num-samples 100` remains valid: it makes 100 sequential single-instance model
calls. The restriction applies to each model call, not to the dataset size.

A separate state API limitation remains: taking a subset of an existing TW
state, such as `state[torch.tensor([1])]` from a two-row batch, then updating it
raises a shape error. Shared event helpers do not consistently select fixed
input tensors through `state.ids`. The single-instance evaluation guard does
not repair this state-slicing API.

The current training loop does not slice its states. It updates the complete
batch and retains finished rows with padding actions. Batched regression tests
exercise differing wait events, capacity reloads, padding, and backpropagation.
Thus this slicing limitation does not affect that training path or sequential
recursive evaluation. Batch shrinking and search algorithms that index states
would need a separate fix and targeted regression tests.

## Validation and next steps

Run the included diagnostic examples from the repository root:

```bash
.venv/bin/python docs/thesis_environment_probes.py
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_environment_integration.py tests/test_problem_costs.py tests/test_location_generation.py -q
```

The original regression subset passed **48 tests**. The diagnostics report
forced-wait, information-set, initialization, and window-generation behavior;
the service-time probe demonstrates the correction described in finding 2,
and initialization now succeeds in all four variants. Regression tests also
check that both recursive modes reject unsupported batch/rollout sizes and
complete valid single-instance evaluations across all four problems.
The full suite passed **479 tests** after adding the explicit recursive
evaluation restriction and its regressions.
They report current behavior and are not permanent tests asserting that these
differences are desirable. Neural probability values illustrate information
dependence in a seeded small model, not a trained checkpoint performance claim.

Before claiming thesis-faithful comparisons, agree which benchmark changes are
intentional. Retain the separation of realized and planning service times and
initial idle waiting in all four variants. Departure commitment has been adopted
for both TW environments. Retain the documented memory-saving masked training
design and distinguish it from recursive online evaluation. A separate
Poisson-arrival evaluation mode can preserve the intentional fixed-size
training and existing benchmarks while testing stochastic workload sizes.
Chapter 7's opening rule is now adopted for newly generated synthetic TW
instances, with both narrow and wide presets. Regenerate synthetic evaluation
files to compare the new setting; loading an old file does not change its windows.
Changes to these semantics should have
targeted regressions and be followed by fresh evaluation of affected baselines
and checkpoints.
