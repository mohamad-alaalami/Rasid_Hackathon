# GoTrain

**An autonomous, configurable ML experimentation agent for geospatial imagery.**  
*AUB ACM × RASID GoTrain Challenge · November 2026*

GoTrain uses **Strands Agents**, **Claude Sonnet on Amazon Bedrock**, and **GPU training on Amazon SageMaker** to inspect GEO-Bench datasets, select compatible models and training pipelines, train them, evaluate on validation data, and iteratively propose the next experiment.

## Why we built it this way

Our first working agent established the core loop surprisingly quickly, achieving approximately **97% validation accuracy** on a classification problem. We could have spent the rest of the hackathon tuning for another fraction of a percentage point. Instead, we focused on a more difficult engineering question:

> **How do we make an autonomous ML engineer configurable, accountable, resource-aware, and reusable across datasets?**

We spent much of the hackathon developing independent controls over what Claude observes, how decisions are recorded, what it may change, and when it needs approval. We then added persistent experimental memory, branching search, compute budgeting, checkpoint continuation, artifact versioning, and an independent evidence verifier.

That choice left less time for comprehensive end-of-hackathon benchmarking. This repository distinguishes **implemented functionality**, **measured validation results**, and **remaining evaluation work** rather than filling missing results with guesses.

## The experimentation loop

```text
GEO-Bench dataset + user constraints
             ↓
Inspect task, channels, classes, candidate models, and resources
             ↓
Claude observes evidence → proposes a model/pipeline and experiment
             ↓
Policy + authority + scope + HITL guards
             ↓
Train on SageMaker GPU → evaluate on VALIDATION split
             ↓
Record metrics, checkpoints, provenance, and resource consumption
             ↓
Independent evidence verification (when enabled)
             ↓
Search memory → refine, branch, continue, revisit, or stop
```

The **test split is not available to the autonomous experimentation loop**. Final test evaluation is a separate, human-authorized, one-time operation for the selected champion of each dataset.

## Configurable operating modes

| Control | Purpose | Choices |
|---|---|---|
| **Feedback (E)** | What evidence Claude receives | `E0`–`E3`, from minimal feedback to richer diagnostics/history |
| **Decision policy (P)** | How Claude records its rationale | `P0`: direct proposal; `P1`: structured evidence, hypothesis, expected signal, and cost justification |
| **Authority (A)** | Which decisions can proceed autonomously | `A1`: proposal/authorization stage; `A2`: bounded autonomous execution; `A0`: reserved, **not implemented** |
| **Scope (S)** | What components Claude may change | `S0`: standard configuration/model choices; `S1`: compositional pipeline changes; `S2`: sandboxed generated components, where supported |
| **HITL** | When human oversight is required | `off`, `critical`, `always` |

These dimensions are **not a single quality ranking**. More information, autonomy, or editable scope may increase cost and risk. For example, `P1 + A2` allows structured explanations while the agent executes permitted experiments autonomously.

**Deployment limitation:** `S2` sandboxed Python execution requires capabilities unavailable in the current SageMaker environment (`unshare` is blocked). It fails closed there; do not interpret this as a successful SageMaker S2 deployment.

## Search, compute, and persistence

**Experiment search graph.** Candidate solutions have persistent identities and parent relationships. Claude can explore alternative branches, refine earlier candidates, and avoid duplicate configurations instead of treating every experiment as a single linear chain. A parent link describes design lineage, **not necessarily reuse of trained weights**.

**Adaptive compute.** The agent can allocate training effort within an experiment budget; continuation support can restore saved optimizer, scheduler, scaler, and random-state information instead of treating every additional training action as a fresh model. The capability is implemented and tested, but individual campaign logs must be inspected to establish whether an *agent-selected continuation* actually occurred.

**Experimental memory.** The search history preserves candidate configurations, parent-child relations, decisions, observations, outcomes, and resource use so later experiments can use the context of earlier work.

**Artifact store.** Local/S3-compatible storage versions training outputs and tracks champions, checkpoint integrity, metrics, provenance, and campaign logs. A model version is an immutable record; a champion is the version selected for final evaluation.

## Independent evidence verification

The verifier is **deterministic and separate from Claude**. It uses saved experiment evidence rather than asking an LLM to judge its own hypotheses. It can:

- Recompute validation diagnostics (classification: macro-F1, balanced accuracy, per-class results and calibration; segmentation: IoU/Dice and class diagnostics).
- Compare *actual* parent/child configuration differences with the prediction recorded by the agent, and flag confounding from simultaneous changes.
- Classify measurable hypotheses as `SUPPORTED`, `CONTRADICTED`, `INCONCLUSIVE`, or `NOT_TESTABLE`.
- Assess uncertainty when sufficient paired or matched-seed data exist; explicitly report insufficient evidence otherwise.
- Track validation reuse, GPU/time/token costs, checkpoint integrity, and score-versus-compute tradeoffs.

This verifier does **not prove causal explanations**, replace GEO-Bench's primary metric, select the final model on the test set, or magically make small observed differences statistically meaningful.

**Verification status:** Implementation passed a local automated test suite (270 passed, 13 skipped in the development report). Its real-artifact integration on SageMaker was **not yet confirmed** in the evidence available for this README. The verifier's read-only agent tools are opt-in.

## Results: measured validation, not final test

The hackathon requires two classification and two segmentation tasks. The following are **best recorded validation metrics** from the available model registry / campaign records; they are **not official final-test scores**.

| Required GEO-Bench problem | Task | Primary metric | Best recorded validation | Confirmed final test |
|---|---|---|---:|---|
| `m-eurosat` | 10-class land-cover classification | Accuracy | **97.90%** | **Not available** |
| `m-pv4ger` | Solar panel classification | Accuracy | **97.90%** (97.8979%, rounded) | **Not available** |
| `m-pv4ger-seg` | Solar panel segmentation | Mean IoU | **95.30%** | **Not available** |
| `m-chesapeake` | 7-class land-cover segmentation | Mean IoU | **69.60%** | **Not available** |

**Do not compare the numeric magnitudes across different datasets or metrics.** In particular, 69.60% mean IoU cannot be judged against 97.90% classification accuracy as though the tasks had the same difficulty.

### Example: real EuroSAT autonomous campaign

A sequential `E2 / P0` campaign completed six GPU experiments:

| Experiment | Change | Best validation accuracy |
|---|---|---:|
| 1 | ResNet-18 baseline | 57.8% |
| 2 | Switch to ConvNeXt-Tiny | 96.8% |
| 3 | Increase input resolution to 128×128 | 97.2% |
| 4 | Augmentation + label smoothing | 97.5% |
| 5 | Increase resolution to 224×224 | **97.9%** |
| 6 | Attempt longer schedule | 97.7% |

The selected model was **experiment 5**, stored as **`m-eurosat/v0017`** (ConvNeXt-Tiny). This campaign recorded **2,941.19 seconds of GPU training** and **389,243 LLM tokens**. One attempted tool action was rejected because it referenced an invalid parent solution; the agent corrected the reference and completed the experiment.

A separate graph-search/adaptive-compute EuroSAT campaign achieved **97.5%** best validation accuracy with approximately **584 seconds** of GPU training. This is an *illustrative quality/compute tradeoff*, **not a controlled head-to-head demonstration**: the campaigns did not have identical GPU budgets. The adaptive campaign completed six initial training actions; do not claim it demonstrated an autonomously chosen resume.

### Decision policy evaluation

A three-seed PV4GER `P0` versus `P1` comparison recorded mean best validation accuracy of approximately **97.47%** and **97.50%**, respectively. This tiny descriptive difference **does not establish that P1 is superior**; compute and the experimental trajectories were not perfectly matched. P1's independently valuable feature is decision traceability.

### Final-test status

The latest available storage summary showed the selected EuroSAT champion as `m-eurosat/v0017`, with test status **`in_progress` and no test score**. The final evaluator may have been interrupted. This is **not a confirmed result**. Do not reset or rerun the one-time evaluator without inspecting its lock/recovery procedure. At that point, the other required tasks also had no verified final-test scores, and `m-pv4ger-seg` had no promoted champion.

## Running GoTrain

The project was developed for the RASID-provided **SageMaker environment**, with Bedrock model access configured there. Install dependencies from the repository's environment/requirements configuration and configure the GEO-Bench dataset location before running training.

Inspect the current CLI flags and guardrails:

```bash
python -m agent.main --help
python -m storage --help
```

Example: a bounded autonomous EuroSAT search campaign:

```bash
python -m agent.main \
  --search --adaptive-compute \
  --feedback E2 --scope S1 --authority A2 \
  --dataset m-eurosat \
  --max-experiments 6 --max-training-seconds 1800
```

**Model registry / storage:**

```bash
python -m storage summary
python -m storage versions m-eurosat
python -m storage promote --help
```

**Independent verifier (after its code is integrated in the environment):**

```bash
python -m pytest -q tests/test_verification.py
python -m verification list --dataset m-forestnet
python -m verification campaign --campaign-id <CAMPAIGN_ID> --feedback E2 --seed 42 --out artifacts/verification_reports
```

**Human-authorized final test — only after freezing the champion and checking any existing evaluation status:**

```bash
python -m storage final-test m-eurosat --yes
```

This command is shown for documentation **only**. It evaluates the registered champion **once per dataset**; it must not be run to choose between competing candidates. For EuroSAT, an `in_progress` marker was already observed, so investigate that state before any further final-test action.

## Evidence and reproducibility

Every credible result should be traceable to its **dataset and split**, model/version, configuration, random seed, primary metric, checkpoint/hash, campaign log, source revision, and measured compute consumption. Local and S3 storage support preserving these records, but availability depends on the configured environment and whether artifacts were actually uploaded.

The source tree contains an agent interface, training/dataset adapters, experiment search and resource accounting, storage commands, and the optional verification subsystem. Refer to the repository itself for the exact file inventory and version-specific usage.

## What remains to validate

- Resolve EuroSAT's existing final-test `in_progress` state safely and report an actual score only if produced by the protected evaluator.
- Freeze and evaluate the selected champions **once** for the remaining required tasks, especially promoting a segmentation champion before attempting final evaluation.
- Run the independent verifier against real persisted SageMaker campaign artifacts and publish those evidence reports.
- Conduct matched-budget, multi-seed comparisons before claiming that one E/P/A/S combination or search strategy is objectively better.

**The project's central choice:** We prioritized a configurable foundation for autonomous experimentation over squeezing marginal improvements from a single benchmark. That explains the breadth of mechanisms implemented—and also why we report evaluation gaps plainly.
