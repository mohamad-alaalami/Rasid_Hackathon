
from strands import Agent
from strands.models import BedrockModel

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    agent_capabilities,
    inspect_dataset,
    inspect_gpu,
    list_eligible_models,
    list_supported_datasets,
    train_registered_model,
)


# ============================================
# CREATE GOTRAIN AI AGENT
# ============================================

def create_agent():

    # Connect to Claude Sonnet through Amazon Bedrock
    model = BedrockModel(
        model_id="eu.anthropic.claude-sonnet-5-5",
        region_name="eu-central-1"
    )

    # Create the agent with its system prompt and tools
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            agent_capabilities,
            inspect_gpu,
            list_supported_datasets,
            list_eligible_models,
            inspect_dataset,
            train_registered_model,
        ]
    )

    return agent


# ============================================
# START GOTRAIN
# ============================================

if __name__ == "__main__":

    print("=" * 50)
    print("Starting GoTrain AI Agent...")
    print("=" * 50)

    # Initialize the agent
    agent = create_agent()

    # Run a bounded end-to-end smoke test through the generic tools.
    agent(
        """
        You are GoTrain, an autonomous geospatial ML agent.

        Complete this workflow in order:

        1. Call agent_capabilities.
        2. Call inspect_gpu.
        3. Call list_supported_datasets.
        4. Call inspect_dataset with dataset_id "m-eurosat".
        5. Call list_eligible_models for "m-eurosat" and select only a model
           whose eligible field is true. Explain the choice using factual metadata.
        6. If those checks succeed, call train_registered_model once for
           "m-eurosat" with the selected model_id, epochs=1, batch_size=32,
           learning_rate=0.0003, weight_decay=0.0001,
           pretrained=true, and seed=42.
        7. Report the task type, primary validation metric, training time,
           checkpoint path, and experiment ID.
        8. Explain whether the run succeeded and recommend one next experiment.

        Use real tool outputs only.
        Never invent metrics and never access the test split.
        Do not run more than one training experiment during this smoke test.
        """
    )

    print("\nGoTrain agent execution completed.")
