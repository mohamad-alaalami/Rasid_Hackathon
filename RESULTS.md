# GoTrain — Results and Evaluation Evidence

**RASID × AUB ACM GoTrain Hackathon | 1 November 2026**

## Summary

GoTrain was developed as a configurable autonomous ML experimentation agent. An early prototype achieved approximately **97% validation accuracy** on a classification problem, after which the team prioritized engineering a configurable experimentation framework rather than pursuing marginal gains on that single task.

The system was exercised through real GPU training runs on SageMaker, including autonomous iterations that selected models, adjusted training configurations, inspected validation metrics, and preserved experiment artifacts.

**Important:** The scores below are **validation scores**, *not* official held-out test scores. At the time these results were collected, the four official test results were not available. Do not report the values below as test performance.

## Four Selected GEO-Bench Problems

| Dataset | Task | Primary metric | Best recorded validation result | Final test result |
|---|---|---|---:|---|
| `m-eurosat` | Land-cover classification | Accuracy | **97.90%** | Not confirmed |
| `m-pv4ger` | Solar-panel classification | Accuracy | **97.90%** (97.8979%) | Not confirmed |
| `m-pv4ger-seg` | Solar-panel segmentation | Mean IoU | **95.30%** (95.2994%) | Not confirmed |
| `m-chesapeake` | Land-cover segmentation | Mean IoU | **69.60%** (69.5982%) | Not confirmed |

Segmentation **mean IoU** and classification **accuracy** are different metrics; these numbers are **not directly comparable**. All four selected tasks are reported, rather than omitting lower-scoring tasks.

## EuroSAT: Recorded Autonomous Sequential Campaign

Dataset: `m-eurosat`  
Campaign ID: `cdede282-8d44-4890-ac41-65d663a3641d`  
Configuration observed: feedback **E2**, policy **P0**, seed **42**, six completed training experiments.

| Experiment | Major configuration or change | Best validation accuracy |
|---|---|---:|
| 1 | Pretrained ResNet-18 RGB baseline | 57.8% |
| 2 | Switch to pretrained ConvNeXt-Tiny RGB | 96.8% |
| 3 | ConvNeXt-Tiny at 128 × 128 resolution | 97.2% |
| 4 | Add flips/90° rotation and label smoothing | 97.5% |
| 5 | Increase resolution to 224 × 224 | **97.9%** |
| 6 | Request a longer schedule (stopped early) | 97.7% |

**Selected checkpoint:** `experiments/exp_20261101_124415_4d9138/best.pt`  
**Registered model version:** `m-eurosat/v0017`  
**Selected model:** pretrained ConvNeXt-Tiny RGB at 224 × 224, with augmentation and label smoothing.

The store summary listed **`m-eurosat/v0017` as champion**, with 0.979 validation accuracy. The same snapshot listed its **test status as `in_progress` and its test score as null**. Therefore no official test accuracy is reported here; the one-time test workflow requires state inspection rather than blindly rerunning.

**Recorded campaign resources:** 2,941.19 seconds of GPU training (~49.0 minutes), 3,025.18 seconds active elapsed time, 15 model invocations, and 389,243 total LLM tokens. Six training experiments completed. One additional proposed action was rejected (`UNKNOWN_PARENT_SOLUTION`) and subsequently corrected. These results demonstrate a working iterative training workflow, not causal proof that any individual configuration change improved performance.

## Additional EuroSAT Graph + Adaptive-Compute Campaign

A separate real GPU campaign exercised solution-search graph functionality and adaptive compute allocation:

| Observed quantity | Value |
|---|---:|
| Best validation accuracy | **97.5%** |
| Completed training actions | 6 |
| GPU training time | ~583.6 seconds |
| LLM tokens | 644,664 |

This campaign produced a strong validation score with substantially less training time than the sequential campaign, but **the comparison was not controlled for compute allocation and other conditions**. It is evidence of implemented functionality and an observed tradeoff, **not** proof that graph/adaptive search is superior. The recorded actions were initial training allocations; an autonomous resume decision was not observed in this campaign.

## P0 versus P1: Preliminary Matched-Seed Validation

A policy comparison on `m-pv4ger` used three matched seeds (43, 44, 45) and three experiments per policy per seed.

| Seed | P0 best validation accuracy | P1 best validation accuracy |
|---|---:|---:|
| 43 | 97.6% | 97.7% |
| 44 | 97.2% | 97.2% |
| 45 | 97.6% | 97.6% |
| **Mean** | **97.47%** | **97.50%** |

The mean difference is approximately **+0.033 percentage points** in favor of P1. This is too small and the sample too limited to establish superiority. Compute budgets and selected configurations were not fully matched, so no causal advantage is claimed. P1 primarily adds structured, auditable decision records.

## ForestNet: Additional Classification Exploration

On the extra `m-forestnet` dataset, an autonomous three-run campaign recorded:

| Candidate | Validation accuracy |
|---|---:|
| Initial multispectral ResNet-18 | 54.70% |
| ResNet-18 with additional regularization/augmentations | **60.16%** |
| RGB ConvNeXt-Tiny | 59.96% |

These are genuine recorded validation outcomes, but this is an **additional exploratory dataset**, not one of the four required submitted tasks. It is included for completeness, not as an official test result or evidence of a controlled S1-versus-S0 advantage.

## What Has and Has Not Been Demonstrated

**Observed in real GPU runs:** Autonomous model/pipeline selection, multiple training iterations, recorded validation outcomes, stored checkpoints/model versions, agent decision traces, resource accounting, and graph/adaptive campaign execution.

**Implemented/tested with limitations:** Structured policy decisions, configurable observation levels, human approval controls, training-state continuation, and independent evidence verification. Their existence does **not** establish that every configuration is empirically superior or that every combination was end-to-end demonstrated.

**Important deployment limitations:** `A0` is reserved rather than a completed operational mode. `S2` sandboxed code generation was not usable on the SageMaker environment due to unavailable Linux isolation support, so it must not be claimed as functioning there. The independent verifier passed local tests, but real-data verification reports were not confirmed in the supplied evidence.

## Required Final-Test Evidence — Not Yet Confirmed

RASID requires **one final test-split score on each of two classification and two segmentation problems**, plus agent logs. The agent is restricted to tuning on validation; the held-out test split is accessed only once after final model selection.

For each task, the final results package should include its *registered champion version*, *final-test status and score*, *checkpoint identity/hash*, *agent/campaign logs*, and *task-specific metric*. **Do not substitute validation results or smoke-test measurements for these final scores.**

### Evidence references from recorded SageMaker outputs

- EuroSAT campaign: `cdede282-8d44-4890-ac41-65d663a3641d` and its `decision_trace` / `resources` output.
- EuroSAT best checkpoint: `experiments/exp_20261101_124415_4d9138/best.pt`.
- EuroSAT champion: `m-eurosat/v0017` (97.9% validation; test `in_progress` in latest observed store snapshot).
- PV4GER champion: `m-pv4ger/v0035` (97.8979% validation).
- PV4GER segmentation best registered version: `m-pv4ger-seg/v0002` (95.2994% validation; **no champion** in latest observed snapshot).
- Chesapeake champion: `m-chesapeake/v0001` (69.5982% validation).
- Official status: `python -m storage summary` (read-only). Do not retry a locked one-time final test without reviewing the evaluator's recovery protocol.

*This document reports only evidence that was visible in the supplied logs and store outputs. It intentionally does not manufacture missing test scores.*
