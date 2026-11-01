# GoTrain

GoTrain is a Strands/Claude agent backed by a registry-driven GEO-Bench training engine.
The same `ml.train_model` API powers terminal runs and agent tool calls.

Registered datasets:

- `m-eurosat`: 10-class classification; 13 bands available, verified RGB baseline selected
- `m-pv4ger`: binary aerial-RGB classification
- `m-pv4ger-seg`: binary aerial-RGB segmentation
- `m-chesapeake`: 7-class aerial RGB+NIR segmentation
- `m-forestnet`: 12-class classification; Landsat-8 Red, Green, Blue, NIR, SWIR1, SWIR2 (uint8, 332x332)
- `m-brick-kiln`: binary classification; 13 stored channels (64x64)
- `m-nz-cattle`: binary aerial-RGB segmentation (500x500)
- `m-cashew-plant`: 7-class segmentation; 12 Sentinel-2 bands plus cloud probability (256x256)

Channels are always selected by verified physical index, never by GEO-Bench's
stored band names. Per [geo-bench #29 / PR #30](https://github.com/ServiceNow/geo-bench/pull/30),
`m-brick-kiln` stores `B01 B02 B03 B04 B05 B07 B8A B08 B11 B12 TCI_R TCI_G TCI_B`
(no B06/B09/B10; the TCI channels are 8-bit true-colour composites) although its
metadata claims the 13 standard Sentinel-2 bands; `m-eurosat` stores B8A last (#28),
which the existing SSL4EO reorder already assumed. Each dataset offers explicit input
views (`rgb`, `multispectral`, `sentinel2_13`, `dataset_channels`); excluded channels
(brick-kiln TCI in the multispectral view, cashew cloud probability) are documented in
`ml/registry.py` and recorded in every checkpoint. The four new datasets are
standardized per channel with statistics from up to 256 evenly spaced training
samples; the statistics are stored in the checkpoint and reused for final test
evaluation. The original four datasets keep their fixed scaling and ImageNet
normalization.

To add another compatible GEO-Bench classification or semantic-segmentation dataset,
add its verified contract to `ml/registry.py`. The loader, model factory, trainer,
evaluator, command line, and agent tools then select their behavior dynamically.

Run a real autonomous E3 campaign on SageMaker:

```bash
python -m agent.main --mode campaign --feedback E3 --max-experiments 5
```

Claude inspects the GPU and registry, selects one dataset, discovers eligible
models, and launches only real `ml.train_model()` experiments. Campaign records
are persisted under `artifacts/campaigns/`. Supply `--campaign-id ID` to resume,
or `--dataset m-pv4ger` to restrict dataset selection.

Run an explicitly bounded one-run debugging smoke test:

```bash
python -m ml.inspect_dataset --dataset-id m-eurosat
python -m ml.run_experiment --dataset-id m-eurosat --config configs/baseline.json
python -m agent.main --mode smoke --dataset m-eurosat --model resnet18_rgb
```

There is no simulated feedback probe. E0-E3 observations are constructed only
from persisted real train/validation experiment artifacts. Missing diagnostics
are reported as unavailable and failures are saved as failures.

## Controlled E0-E3 comparison

Run a small real SageMaker pilot (four isolated campaigns, three experiments each):

```bash
python -m agent.compare_feedback \
  --dataset m-pv4ger \
  --repeats 1 \
  --max-experiments 3 \
  --seed 42
```

Run a larger matched-seed comparison:

```bash
python -m agent.compare_feedback \
  --dataset m-pv4ger \
  --repeats 3 \
  --max-experiments 3 \
  --seed 42
```

Every mode receives the same forced first real configuration for a given seed.
E0/E1 have no history tool, E2 has isolated campaign history, and E3 additionally
has real recorded diagnostics. Outputs are written to a new timestamped directory
under `artifacts/feedback_comparison/` as JSON, CSV, Markdown, and per-trial decision
traces; previous campaign and comparison results are never overwritten.

## Experiment-selection policies P0/P1

`--policy` is independent of `--feedback`:

- `P0` (default): free selection; the agent, prompt, and training tool are unchanged.
- `P1`: evidence-grounded selection. `run_campaign_experiment` additionally requires a
  `decision_record` with nonempty `observation`, `action`, `expected_signal`, and
  `cost_justification`. Malformed records are rejected with `INVALID_DECISION_RECORD`
  before reservation, so they never train or consume budget. The record never
  overrides the deterministic model, batch, duplicate, and budget checks.

```bash
python -m agent.main --mode campaign --feedback E2 --policy P1 --dataset m-pv4ger --max-experiments 4
```

Every attempt (accepted, failed, or rejected) is appended to
`artifacts/campaigns/decisions.jsonl` with its policy, campaign, configuration,
decision record, and the previous/incumbent experiment at decision time. No agent
tool reads that log, so P1 gets no extra observations. The final report joins it with
the measured results into `decision_trace`, keeping Claude's `expected_signal` apart
from the measured `validation_score`. A campaign ID cannot switch policies.

Compare P0 and P1 under frozen common conditions (same E-profile, dataset, seed,
forced first run, model registry, evaluator, and experiment budget; separate campaigns):

```bash
python -m agent.compare_policies --dataset m-pv4ger --feedback E2 --repeats 3 --max-experiments 4 --seed 42
```

Reports go to `artifacts/policy_comparison/<run_id>/`. They are descriptive: no winner
is claimed, and one campaign per policy is not statistical evidence. Every run trains
from the registry's pretrained weights, because checkpoint resumption is unsupported.
GPU time is measured and reported but not enforced as a budget.

## Resource accounting

Every `agent.main` campaign (and `agent.compare_policies` trial) records one session per
agent process in `artifacts/campaigns/resources.jsonl`: provider-reported Bedrock tokens,
model invocations (Strands event-loop cycles), LLM latency, and elapsed time. Training
time is read from the existing experiment records (`training_seconds`, else
`runtime_seconds`; never both). Values that were not reported are `unknown`, never zero.
Costs need an explicit price file; copy `configs/prices.example.json` and fill it in.

```bash
python -m research.resource_accounting --campaign-id ID --prices configs/prices.json
python -m research.resource_accounting --campaign-id ID --json
```

## Composable pipelines, solutions, and autonomy

`ml/components.py` defines every supported training component with its parameters and
compatibility constraints: input view and normalization, spatial augmentations
(`hflip`, `vflip`, `rot90`, `random_resized_crop`; masks move with images), photometric
augmentations (`brightness`, `contrast`, `gaussian_noise`; one shared factor across all
channels, so band relationships are preserved), losses (`cross_entropy` with label
smoothing, `focal`, segmentation-only `dice_cross_entropy`, optional inverse-frequency
class weights), optimizers (`adamw`, `sgd`), schedulers (`cosine`, `constant`, `step`,
`warmup_cosine`), and classification-head `dropout`. A `pipeline` object in the training
config selects them; omitted keys keep the historical defaults (flips, cross-entropy,
AdamW, cosine), so existing campaigns are unchanged. Incompatible combinations are
rejected with every reason listed.

```bash
python -m ml.smoke_test --dataset-id m-cashew-plant --model unet_r34 \
  --pipeline '{"loss": {"name": "dice_cross_entropy", "class_weighting": "inverse_frequency"}, "augmentations": [{"name": "rot90"}, {"name": "hflip"}]}'
```

Every training run is recorded against an immutable, content-addressed `SolutionSpec`
under `artifacts/solutions/<solution_id>/` (`$GOTRAIN_SOLUTIONS_DIR` overrides):
`spec.json` (dataset contract hash, input channels, normalization policy, model and
pretrained provenance, pipeline, training settings, seed) and an append-only
`events.jsonl` with proposals (parent solution, change description, campaign, autonomy)
and executions (validation results, resources, checkpoint SHA-256, environment, dataset
fingerprint). `initialization` and `custom_components` are reserved for later warm starts
and sandboxed code.

Autonomy has two independent axes, separate from P0/P1 and HITL:
`--authority A2` (default: Claude chooses and launches) or `A1` (launches must first be
proposed with `propose_solution`), and `--scope S0` (default: existing knobs) or `S1`
(compose pipeline components via `list_pipeline_components`, a `pipeline` argument,
`parent_solution_id`/`change_description`, and `inspect_solutions`). A0 (system
decisions) and S2 (sandboxed code) are reserved.

```bash
python -m agent.main --dataset m-forestnet --feedback E2 --scope S1 --max-experiments 3
```

## Human-in-the-loop (`--hitl`)

- `off` (default): fully autonomous; agent, prompt, and tools are unchanged.
- `always`: before every training run Claude calls `request_human_consultation` with the
  evidence, options (with exact configurations and measured-cost estimates), and its
  recommendation. The human picks an option, delegates back to Claude, or stops.
- `critical`: autonomous, but approval is required after `--hitl-failure-threshold`
  consecutive failed runs, or when a run's measured-cost estimate exceeds
  `--hitl-cost-threshold-usd`. Claude must consult before proposing code changes and may
  consult when evidence is insufficient.

With a cost threshold, a run whose cost cannot be estimated from measured runs of the
same model asks the human by default (`--hitl-unknown-cost allow` opts out).

The training tool checks approval before reservation, so blocked launches use no budget.
A selected configuration must be launched exactly. A consultation raises a Strands
interrupt: the agent loop stops (HITL agents run tools sequentially), the conversation
and interrupt are saved by Strands' `FileSessionManager`, and no training job is running.
By default the process exits; the human answers by request number and the campaign
resumes once. `--hitl-interactive` instead asks on the terminal between agent calls.

```bash
python -m agent.main --hitl always --dataset m-pv4ger --feedback E2 --max-experiments 4 --campaign-id demo1
python -m agent.hitl show --campaign-id demo1
python -m agent.hitl respond --campaign-id demo1 --request-id 1 --choice A --note "keep native resolution"
python -m agent.main --resume --campaign-id demo1
python -m agent.main --hitl critical --dataset m-chesapeake --hitl-cost-threshold-usd 2 --prices configs/prices.json
python -m agent.main --hitl always --hitl-interactive --dataset m-pv4ger
```

Run the Phase-A one-batch GPU checks before adding model architectures:

```bash
python -m ml.smoke_test --dataset-id m-pv4ger --model resnet18_rgb
python -m ml.smoke_test --dataset-id m-pv4ger-seg --model fcn_resnet50
python -m ml.smoke_test --dataset-id m-chesapeake --model fcn_resnet50
```

Smoke tests and training preserve each dataset's verified native resolution by
default: EuroSAT 64x64, PV4GER/PV4GER-Seg 320x320, and Chesapeake 256x256.
Use `--image-size N` only as an explicit square resize experiment (for example,
after a native-resolution GPU out-of-memory result); the image uses bilinear
interpolation and segmentation masks use nearest-neighbor interpolation.

Training uses only `train` and `valid`. The test split is reserved for one final evaluation.

## Validation-based early stopping

Early stopping is optional and disabled by default so existing baselines remain
unchanged. Enable it in any training configuration or agent tool call:

```json
{
  "epochs": 15,
  "early_stopping": true,
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.001
}
```

`epochs` is then the maximum. Training stops after the primary validation metric
(accuracy for classification or IoU for segmentation) fails to improve by more
than `early_stopping_min_delta` for `early_stopping_patience` consecutive epochs.
The best real checkpoint is preserved. Results record requested/completed epochs,
whether stopping occurred, and the factual stop reason.

## Model selection

`ml/model_registry.py` is the factual source for model compatibility and provenance.
Claude must call `list_eligible_models(dataset_id)` and may select only entries whose
runtime `eligible` value is true. Missing packages or exact checkpoints are reported;
the engine never substitutes random weights.

Target configurations:

- `resnet18_rgb`: TorchVision `ResNet18_Weights.IMAGENET1K_V1`
- `convnext_tiny_rgb`: TorchVision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- `resnet18_s2`: SSL4EO-S12 `B13_rn18_moco_0099_ckpt.pth`
- `resnet50_s2`: SSL4EO-S12 `B13_rn50_dino_0099.pth`
- `unet_r34`: SMP U-Net with pinned ResNet-34 ImageNet encoder provenance
- `deeplabv3plus_r50`: SMP DeepLabV3+ with pinned ResNet-50 ImageNet encoder provenance
- `resnet18_ms`, `resnet50_ms`: TorchVision ImageNet ResNets for multispectral views

Compatibility is a capability rule, not a dataset list: a model is offered when its
task type matches and the dataset declares the input view it consumes. When a model
takes more than three channels, its pretrained RGB filters are placed on the channels
verified as red/green/blue and every other channel starts from the mean pretrained RGB
filter (documented adaptation, never random). U-Net and DeepLabV3+ pad inputs to a
multiple of 32 and crop outputs back, so `m-nz-cattle` trains at native 500x500.

The existing `fcn_resnet50` implementation is retained as a legacy segmentation
baseline but is not presented as one of the requested target configurations.

Run every currently eligible combination on SageMaker:

```bash
python -m ml.smoke_matrix --include-legacy
```

The command saves factual runtime and peak-memory results to
`artifacts/smoke_matrix.json`. Models with missing dependencies or checkpoints are
recorded as skipped, not silently initialized.

## Dataset and model-version store

GoTrain can keep datasets, trained model versions, and agent logs in Amazon S3
(or any shared directory) instead of only the notebook folder. It is disabled
unless `GOTRAIN_STORE` is set; without it, training, tools, and prompts are unchanged.

```bash
export GOTRAIN_STORE=s3://<bucket>/gotrain   # or a directory such as ~/gotrain-store
python -m storage doctor                     # checks access; suggests the SageMaker default bucket
```

When enabled:

- Every completed training run is uploaded automatically as the next numbered
  version of its dataset (`m-eurosat/v0003`): checkpoint, config, metrics, and a
  provenance record with the validation score, experiment ID, and git commit.
  Upload problems are reported in the result and never fail a finished run.
- Campaign agents get `model_store_status`, `sync_dataset` (download a dataset
  if this instance lacks it, upload it if the store lacks it) and
  `upload_experiment`. E2/E3 campaigns also get `list_model_versions`,
  `get_model_version`, `promote_model_version`, and `download_model_version`;
  E0/E1 do not, because version history would break their no-history contract.
  The controlled comparison runners never receive store tools.

Store layout:

```text
datasets/<benchmark>/<dataset>.tar (+ .manifest.json with SHA-256)
models/<dataset>/v0001/{version.json, best.pt, config.json, metrics.json, training_result.json}
models/<dataset>/champion.json          current champion + promotion history
models/<dataset>/test_evaluation.json   the one-time final test result
logs/{artifacts,experiments}/...        campaign logs and experiment records
```

Human commands (`python -m storage <command>`):

```bash
python -m storage push-dataset m-eurosat          # pull-dataset on another instance
python -m storage register --all                  # version experiments trained before the store
python -m storage versions m-eurosat
python -m storage promote m-eurosat 3 --reason "best validation accuracy"
python -m storage final-test m-eurosat --yes      # champion only, once per dataset
python -m storage summary                         # champion, validation and test score per dataset
python -m storage push-logs
```

`final-test` is the only path to the test split. It evaluates the champion with
the exact preprocessing stored in its checkpoint, records the result, and locks the
dataset: a second test evaluation or a later promotion is refused. If a run crashes
before reporting metrics, retry with `--retry-unfinished`.
