
import json

from strands import Agent
from strands.models import BedrockModel

from agent.feedback import build_feedback
from agent.prompts import SYSTEM_PROMPT
from agent.tools import make_diagnostics_tool


def main():

    # SIMULATED TRAINING RESULTS — NOT REAL TRAINING
    experiment = {
        "experiment_id": 1,
        "status": "completed",
        "evaluation_split": "validation",
        "configuration": {
            "architecture": "resnet18",
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 32
        },
        "validation_score": 0.78,
        "train_loss": [
            0.95, 0.72, 0.51, 0.35, 0.22
        ],
        "val_loss": [
            1.01, 0.82, 0.65, 0.61, 0.69
        ],
        "validation_metric_curve": [
            0.55, 0.64, 0.72, 0.78, 0.76
        ],
        "detailed_diagnostics": {
            "per_class_metrics": {
                "forest": 0.89,
                "urban": 0.62,
                "agriculture": 0.81
            },
            "learning_rate_history": [
                0.001, 0.001, 0.001, 0.001, 0.001
            ]
        }
    }

    # Previous simulated experiment
    history = [
        {
            "experiment_id": 0,
            "configuration": {
                "architecture": "small_cnn",
                "learning_rate": 0.001,
                "epochs": 5,
                "batch_size": 32
            },
            "validation_score": 0.74,
            "status": "completed"
        }
    ]

    # SELECT E3
    version = "E3"

    feedback = build_feedback(
        version=version,
        experiment=experiment,
        remaining_budget=3,
        history=history
    )

    print("\nFEEDBACK SENT TO CLAUDE:")
    print(json.dumps(feedback, indent=2))

    # Register the adaptive tool for E3 only
    tools = []

    if version == "E3":
        tools.append(
            make_diagnostics_tool(experiment)
        )

    model = BedrockModel(
        model_id="eu.anthropic.claude-sonnet-5-5",
        region_name="eu-central-1"
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools
    )

    prompt = f"""
    You are GoTrain.

    This is a simulated feedback experiment.
    No real training is being performed.

    Dataset: m-eurosat
    Task: classification
    Feedback version: {version}

    FEEDBACK:
    {json.dumps(feedback, indent=2)}

    Allowed architectures:
    - resnet18
    - small_cnn

    Allowed learning rates:
    - 0.01
    - 0.001
    - 0.0001

    Allowed batch sizes:
    - 16
    - 32

    Allowed epochs:
    - 5
    - 10

    ADAPTIVE DIAGNOSTICS:

    You have access to request_diagnostics.

    Decide independently whether the current feedback
    is sufficient to propose a useful improvement.

    If additional diagnostics would materially help,
    call request_diagnostics for the information needed.

    You may request up to two diagnostics.

    Do not request diagnostics unnecessarily.

    If the available feedback is sufficient,
    proceed directly to proposing the next configuration.

    Explain why you requested or skipped extra diagnostics.

    NEXT:
    1. Analyze the initial feedback.
    2. If you requested diagnostics, analyze them.
       Otherwise, explain why they were unnecessary.
    3. Explain whether the extra information
       changed your reasoning.
    4. Propose ONE new training configuration.
    5. Change only one parameter.
    6. Explain your decision.

    Do not invent results.
    Do not claim to have trained a model.
    Do not request test-set data.
    """

    print("\nCLAUDE'S E3 DECISION:\n")

    agent(prompt)


if __name__ == "__main__":
    main()
