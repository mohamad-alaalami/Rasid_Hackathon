
from strands import Agent
from strands.models import BedrockModel

from agent.prompts import SYSTEM_PROMPT
from agent.tools import agent_capabilities, inspect_gpu


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
            inspect_gpu
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

    # Ask Claude to use both tools
    agent(
        """
        You are GoTrain, an autonomous geospatial ML agent.

        First, call the agent_capabilities tool.

        Second, call the inspect_gpu tool.

        Then report:
        1. Your currently available capabilities.
        2. The GPU name.
        3. Total GPU memory.
        4. Free GPU memory.
        5. CUDA version.
        6. Which ML features still need implementation.

        Use real tool outputs only.
        Do not invent any information.
        """
    )

    print("\nGoTrain agent execution completed.")
