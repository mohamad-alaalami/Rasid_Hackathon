
# agent/experiments.py

"""
GoTrain experiment persistence and budget control.

- Isolated campaigns
- Persistent experiment budgets
- Duplicate configuration prevention
- Concurrent-request protection
- Validation-only experiment comparison
"""

import json
import math
import os
import threading
import uuid

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_VERSIONS = ("E0", "E1", "E2", "E3")
RESULTS_DIR = Path("artifacts/campaigns")
_PROCESS_LOCK = threading.RLock()

try:
    import fcntl
except ImportError:  # Windows development; SageMaker uses the fcntl path below.
    fcntl = None


def create_campaign_id():
    """Create an identifier for a new evaluation campaign."""
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _check_version(version):
    if version not in FEEDBACK_VERSIONS:
        raise ValueError("Invalid feedback version.")


def _check_identity(campaign_id, dataset, seed):
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise ValueError("A nonempty campaign_id is required.")

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("A nonempty dataset is required.")

    if type(seed) is not int:
        raise ValueError("Seed must be an integer.")


def _finite_number(value, allow_zero=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value >= 0 if allow_zero else value > 0)
    )


def _configuration_key(configuration):
    return json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def validate_configuration(configuration, allowed_values=None):
    """Validate basic training parameters and optional choices."""

    if not isinstance(configuration, dict):
        return "Configuration must be a dictionary."

    required = (
        "architecture",
        "learning_rate",
        "epochs",
        "batch_size",
    )

    if any(name not in configuration for name in required):
        return "Missing required training parameters."

    if (
        not isinstance(configuration["architecture"], str)
        or not configuration["architecture"].strip()
    ):
        return "Invalid architecture."

    if not _finite_number(configuration["learning_rate"]):
        return "Learning rate must be positive and finite."

    for name in ("epochs", "batch_size"):
        value = configuration[name]
        if type(value) is not int or value <= 0:
            return f"{name} must be a positive integer."

    if allowed_values is not None:
        for name, value in configuration.items():
            if name not in allowed_values:
                return f"Unsupported parameter: {name}"

            if value not in allowed_values[name]:
                return f"Unsupported value for {name}: {value}"

    try:
        _configuration_key(configuration)
    except (TypeError, ValueError):
        return "Configuration must contain valid JSON values."

    return None


@contextmanager
def _locked_storage():
    """
    Serialize reads and writes across processes
    sharing this SageMaker filesystem.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lock_path = RESULTS_DIR / ".experiments.lock"

    with _PROCESS_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _path(version):
    return RESULTS_DIR / f"{version}.jsonl"


def _read_events(version):
    path = _path(version)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def _append_event(version, record):
    """
    Append and flush an event to persistent storage.
    Must be called while holding _locked_storage().
    """
    with _path(version).open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, allow_nan=False) + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def _latest_records(version):
    """
    Collapse reservation and final-result events
    into one current record per experiment.
    """
    latest = {}

    for record in _read_events(version):
        key = (
            record.get("campaign_id"),
            record.get("dataset"),
            record.get("seed"),
            str(record.get("experiment_id")),
        )
        latest[key] = record

    return sorted(
        latest.values(),
        key=lambda run: (
            run.get("started_at", run.get("recorded_at", "")),
            str(run.get("experiment_id")),
        ),
    )


def _scope(version, campaign_id, dataset, seed):
    return [
        run
        for run in _latest_records(version)
        if run.get("campaign_id") == campaign_id
        and run.get("dataset") == dataset
        and run.get("seed") == seed
    ]


def _valid_final_record(experiment):
    if experiment.get("evaluation_split") != "validation":
        raise ValueError("Only validation experiments are allowed.")

    if experiment.get("status") not in ("completed", "failed"):
        raise ValueError("Invalid final experiment status.")

    runtime = experiment.get("runtime_seconds")

    if not _finite_number(runtime, allow_zero=True):
        raise ValueError("Runtime must be nonnegative and finite.")

    if experiment["status"] == "completed":
        if not _finite_number(
            experiment.get("validation_score"),
            allow_zero=True,
        ) and not (
            isinstance(experiment.get("validation_score"), (int, float))
            and not isinstance(experiment.get("validation_score"), bool)
            and math.isfinite(experiment["validation_score"])
        ):
            raise ValueError(
                "Completed experiments require a finite validation score."
            )

    if any(key.lower().startswith("test") for key in experiment):
        raise ValueError("Test-set results cannot be saved here.")


def save_experiment(version, experiment):
    """
    Save one independently produced completed/failed result.

    campaign_id is required inside experiment.
    For guarded training, use reserve_training()
    followed by finalize_training() instead.
    """
    _check_version(version)

    required = (
        "campaign_id",
        "dataset",
        "seed",
        "experiment_id",
        "configuration",
        "status",
        "runtime_seconds",
        "evaluation_split",
    )

    for field in required:
        if field not in experiment:
            raise ValueError(f"Missing field: {field}")

    _check_identity(
        experiment["campaign_id"],
        experiment["dataset"],
        experiment["seed"],
    )
    _valid_final_record(experiment)

    error = validate_configuration(experiment["configuration"])
    if error:
        raise ValueError(error)

    with _locked_storage():
        previous = _scope(
            version,
            experiment["campaign_id"],
            experiment["dataset"],
            experiment["seed"],
        )

        if any(
            str(run["experiment_id"])
            == str(experiment["experiment_id"])
            for run in previous
        ):
            raise ValueError(
                "Experiment ID already exists. "
                "Use finalize_training() for reservations."
            )

        config_key = _configuration_key(experiment["configuration"])

        if any(
            _configuration_key(run["configuration"]) == config_key
            for run in previous
        ):
            raise ValueError("Duplicate training configuration.")

        timestamp = _now()
        record = {
            **experiment,
            "feedback_version": version,
            "started_at": timestamp,
            "recorded_at": timestamp,
        }

        _append_event(version, record)

    return str(_path(version))


def load_experiments(
    version,
    campaign_id=None,
    dataset=None,
    seed=None,
):
    """Load current records, optionally filtered by trial."""

    _check_version(version)

    with _locked_storage():
        records = _latest_records(version)

    if campaign_id is not None:
        records = [
            run for run in records
            if run.get("campaign_id") == campaign_id
        ]

    if dataset is not None:
        records = [
            run for run in records
            if run.get("dataset") == dataset
        ]

    if seed is not None:
        records = [
            run for run in records
            if run.get("seed") == seed
        ]

    return records


def load_campaign_experiments(version, campaign_id, seed=None):
    """Load all real experiment records belonging to one campaign."""
    return load_experiments(
        version,
        campaign_id=campaign_id,
        seed=seed,
    )


def _budget_usage(runs):
    """Count all launches and reserved/used GPU seconds."""

    gpu_seconds = 0.0

    for run in runs:
        if run["status"] == "launched":
            gpu_seconds += run.get("_reserved_gpu_seconds", 0) or 0
        else:
            gpu_seconds += run.get("runtime_seconds", 0) or 0

    return len(runs), gpu_seconds


def get_budget_status(
    version,
    campaign_id,
    dataset,
    seed,
    max_experiments=3,
    max_gpu_seconds=None,
):
    """Read persistent budget usage for one isolated trial."""

    _check_version(version)
    _check_identity(campaign_id, dataset, seed)

    with _locked_storage():
        runs = [
            run for run in _latest_records(version)
            if run.get("campaign_id") == campaign_id and run.get("seed") == seed
        ]
    if runs and "_max_experiments" in runs[0]:
        max_experiments = runs[0]["_max_experiments"]
        max_gpu_seconds = runs[0].get("_max_gpu_seconds")

    launched, gpu_used = _budget_usage(runs)

    return {
        "campaign_id": campaign_id,
        "feedback_version": version,
        "launched_experiments": launched,
        "max_experiments": max_experiments,
        "remaining_experiments": max(
            0, max_experiments - launched
        ),
        "gpu_seconds_used_or_reserved": gpu_used,
        "max_gpu_seconds": max_gpu_seconds,
        "remaining_gpu_seconds": (
            None if max_gpu_seconds is None
            else max(0.0, max_gpu_seconds - gpu_used)
        ),
    }


def reserve_training(
    version,
    campaign_id,
    dataset,
    seed,
    configuration,
    max_experiments=3,
    max_gpu_seconds=None,
    requested_gpu_seconds=None,
    allowed_values=None,
    admit=None,
):
    """
    Atomically reserve a training attempt BEFORE GPU use.

    A reservation consumes one experiment slot even
    when training later fails or the process restarts.
    ``admit(campaign_runs)`` is an optional extra check evaluated under the
    same lock (for example "this continuation still starts from the latest
    state"); it returns a rejection dict or None. It must not take the lock.
    """
    _check_version(version)
    _check_identity(campaign_id, dataset, seed)

    if type(max_experiments) is not int or max_experiments < 1:
        raise ValueError("max_experiments must be positive.")

    if max_gpu_seconds is not None and not _finite_number(
        max_gpu_seconds
    ):
        raise ValueError("max_gpu_seconds must be positive.")

    error = validate_configuration(configuration, allowed_values)

    if error:
        return {
            "status": "rejected",
            "error_code": "INVALID_CONFIGURATION",
            "message": error,
        }

    with _locked_storage():
        runs = _scope(version, campaign_id, dataset, seed)
        campaign_runs = [
            run for run in _latest_records(version)
            if run.get("campaign_id") == campaign_id and run.get("seed") == seed
        ]
        selected_datasets = {run.get("dataset") for run in campaign_runs}
        if selected_datasets and dataset not in selected_datasets:
            return {
                "status": "rejected",
                "error_code": "CAMPAIGN_DATASET_LOCKED",
                "message": (
                    "A campaign may optimize only one dataset; it is already locked "
                    f"to {sorted(selected_datasets)}."
                ),
            }

        # Do not allow the configured limits to change
        # after a campaign trial has started.
        for run in runs:
            if "_max_experiments" in run and (
                run["_max_experiments"] != max_experiments
                or run.get("_max_gpu_seconds") != max_gpu_seconds
            ):
                return {
                    "status": "rejected",
                    "error_code": "BUDGET_CONFIG_MISMATCH",
                    "message": "Budget limits changed during the trial.",
                }

        config_key = _configuration_key(configuration)

        for run in runs:
            if _configuration_key(run["configuration"]) == config_key:
                return {
                    "status": "rejected",
                    "error_code": "DUPLICATE_CONFIGURATION",
                    "message": "This configuration was already launched.",
                    "existing_experiment_id": run["experiment_id"],
                }

        if admit is not None:
            refused = admit(campaign_runs)
            if refused:
                return refused

        launched, gpu_used = _budget_usage(campaign_runs)

        if launched >= max_experiments:
            return {
                "status": "rejected",
                "error_code": "BUDGET_EXHAUSTED",
                "remaining_experiments": 0,
            }

        reserved_seconds = 0.0

        if max_gpu_seconds is not None:
            available = max_gpu_seconds - gpu_used

            if available <= 0:
                return {
                    "status": "rejected",
                    "error_code": "BUDGET_EXHAUSTED",
                    "message": "GPU-time budget exhausted.",
                }

            # If no runtime is requested, reserve all
            # remaining GPU seconds for this attempt.
            if requested_gpu_seconds is None:
                requested_gpu_seconds = available

            if (
                not _finite_number(requested_gpu_seconds)
                or requested_gpu_seconds > available
            ):
                return {
                    "status": "rejected",
                    "error_code": "BUDGET_EXHAUSTED",
                    "message": "Insufficient GPU-time budget.",
                }

            reserved_seconds = float(requested_gpu_seconds)

        numeric_ids = [
            run["experiment_id"]
            for run in runs
            if type(run["experiment_id"]) is int
        ]

        experiment_id = max(numeric_ids, default=0) + 1

        timestamp = _now()

        record = {
            "campaign_id": campaign_id,
            "feedback_version": version,
            "dataset": dataset,
            "seed": seed,
            "experiment_id": experiment_id,
            "configuration": configuration,
            "evaluation_split": "validation",
            "validation_score": None,
            "status": "launched",
            "runtime_seconds": 0.0,
            "_max_experiments": max_experiments,
            "_max_gpu_seconds": max_gpu_seconds,
            "_reserved_gpu_seconds": reserved_seconds,
            "started_at": timestamp,
            "recorded_at": timestamp,
        }

        _append_event(version, record)

        return {
            "status": "reserved",
            "experiment_id": experiment_id,
            "campaign_id": campaign_id,
            "remaining_experiments": max_experiments - launched - 1,
            "max_runtime_seconds": (
                reserved_seconds
                if max_gpu_seconds is not None
                else None
            ),
        }


def finalize_training(
    version,
    campaign_id,
    dataset,
    seed,
    experiment_id,
    result,
):
    """
    Finalize a reserved training run.

    A failed run remains recorded and consumes
    its previously reserved experiment slot.
    """
    _check_version(version)
    _check_identity(campaign_id, dataset, seed)

    if not isinstance(result, dict):
        raise ValueError("Training result must be a dictionary.")

    immutable = {
        "campaign_id",
        "feedback_version",
        "dataset",
        "seed",
        "experiment_id",
        "configuration",
        "started_at",
        "_max_experiments",
        "_max_gpu_seconds",
        "_reserved_gpu_seconds",
    }

    if any(key in immutable for key in result):
        raise ValueError("Training result contains reserved fields.")

    if result.get("evaluation_split") != "validation":
        raise ValueError("Training must report validation results.")

    if any(key.lower().startswith("test") for key in result):
        raise ValueError("Test results are prohibited.")

    with _locked_storage():
        runs = _scope(version, campaign_id, dataset, seed)

        previous = next(
            (
                run for run in runs
                if str(run["experiment_id"])
                == str(experiment_id)
            ),
            None,
        )

        if previous is None:
            raise ValueError("Unknown experiment reservation.")

        if previous["status"] != "launched":
            return previous

        record = {
            **previous,
            **result,
            "recorded_at": _now(),
        }

        _valid_final_record(record)
        _append_event(version, record)

        return record


def get_best_experiment(
    version,
    campaign_id,
    dataset,
    seed,
    higher_is_better=True,
):
    """Find the best completed validation experiment."""

    runs = load_experiments(
        version,
        campaign_id=campaign_id,
        dataset=dataset,
        seed=seed,
    )

    completed = [
        run for run in runs
        if run["status"] == "completed"
    ]

    if not completed:
        return None

    selector = max if higher_is_better else min

    return selector(
        completed,
        key=lambda run: run["validation_score"],
    )


def compare_versions(
    dataset,
    seed,
    experiments_per_version,
    higher_is_better=True,
    *,
    campaign_id,
):
    """
    Compare feedback strategies inside one campaign.

    Winner is selected only when all versions have
    the required number of finalized attempts.
    """
    _check_identity(campaign_id, dataset, seed)

    if (
        type(experiments_per_version) is not int
        or experiments_per_version < 1
    ):
        raise ValueError("Experiment budget must be positive.")

    summary = {}
    ready = True

    # Read every version under one consistent lock.
    with _locked_storage():
        all_runs = {
            version: _scope(
                version, campaign_id, dataset, seed
            )
            for version in FEEDBACK_VERSIONS
        }

    for version, runs in all_runs.items():
        completed = [
            run for run in runs
            if run["status"] == "completed"
        ]
        failed = [
            run for run in runs
            if run["status"] == "failed"
        ]
        launched = [
            run for run in runs
            if run["status"] == "launched"
        ]

        if (
            len(runs) != experiments_per_version
            or launched
        ):
            ready = False

        best = None

        if completed:
            selector = max if higher_is_better else min
            best = selector(
                completed,
                key=lambda run: run["validation_score"],
            )

        configurations = [
            _configuration_key(run["configuration"])
            for run in runs
        ]

        summary[version] = {
            "experiments": len(runs),
            "successful": len(completed),
            "failed": len(failed),
            "in_progress": len(launched),
            "duplicate_configurations": (
                len(configurations)
                - len(set(configurations))
            ),
            "total_runtime_seconds": sum(
                run.get("runtime_seconds", 0)
                for run in runs
            ),
            "best_validation_score": (
                best["validation_score"] if best else None
            ),
            "best_experiment_id": (
                best["experiment_id"] if best else None
            ),
        }

    winner = None

    if ready:
        candidates = [
            version
            for version in FEEDBACK_VERSIONS
            if summary[version]["best_validation_score"] is not None
        ]

        if candidates:
            selector = max if higher_is_better else min

            winner = selector(
                candidates,
                key=lambda version: summary[version][
                    "best_validation_score"
                ],
            )

    return {
        "campaign_id": campaign_id,
        "dataset": dataset,
        "seed": seed,
        "experiments_per_version": experiments_per_version,
        "comparison_ready": ready,
        "versions": summary,
        "winner": winner,
    }
