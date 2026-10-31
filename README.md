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
