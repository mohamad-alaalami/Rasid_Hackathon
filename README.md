# GoTrain

GoTrain is a Strands/Claude agent backed by a registry-driven GEO-Bench training engine.
The same `ml.train_model` API powers terminal runs and agent tool calls.

Registered datasets:

- `m-eurosat`: 10-class classification; 13 bands available, verified RGB baseline selected
- `m-pv4ger`: binary aerial-RGB classification
- `m-pv4ger-seg`: binary aerial-RGB segmentation
- `m-chesapeake`: 7-class aerial RGB+NIR segmentation

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
