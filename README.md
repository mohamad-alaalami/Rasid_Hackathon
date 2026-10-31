# GoTrain

GoTrain is a Strands/Claude agent backed by a registry-driven GEO-Bench training engine.
The same `ml.train_model` API powers terminal runs and agent tool calls.

Registered datasets:

- `m-eurosat`: 10-class classification using all 13 Sentinel-2 channels
- `m-pv4ger`: binary aerial-RGB classification
- `m-pv4ger-seg`: binary aerial-RGB segmentation
- `m-chesapeake`: 7-class aerial RGB+NIR segmentation

To add another compatible GEO-Bench classification or semantic-segmentation dataset,
add its verified contract to `ml/registry.py`. The loader, model factory, trainer,
evaluator, command line, and agent tools then select their behavior dynamically.

Run a smoke test:

```bash
python -m ml.inspect_dataset --dataset-id m-eurosat
python -m ml.run_experiment --dataset-id m-eurosat --config configs/baseline.json
python -m agent.main
```

Training uses only `train` and `valid`. The test split is reserved for one final evaluation.
