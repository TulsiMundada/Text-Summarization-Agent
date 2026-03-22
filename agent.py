import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext

# --- Setup Logging and Environment ---

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

# --- TOOL: Save user input ---

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's input text to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}


# --- 1. Summarizer Agent ---

summarizer_agent = Agent(
    name="summarizer_agent",
    model=model_name,
    description="Summarizes the given input text.",
    instruction="""
    You are a helpful AI assistant.

    Your task is to summarize the given text clearly and concisely.

    - Keep the summary short
    - Preserve key meaning
    - Avoid unnecessary details

    TEXT:
    { PROMPT }
    """,
    output_key="summary_output"
)

# --- 2. Formatter Agent (optional but keeps your structure clean) ---

response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Formats the summary into a clean response.",
    instruction="""
    You are a helpful assistant.

    Take the SUMMARY_OUTPUT and present it cleanly.

    Keep it short and readable.

    SUMMARY:
    { summary_output }
    """
)

# --- Workflow ---

summarization_workflow = SequentialAgent(
    name="summarization_workflow",
    description="Workflow to summarize user input text.",
    sub_agents=[
        summarizer_agent,
        response_formatter,
    ]
)

# --- Root Agent ---

root_agent = Agent(
    name="summarization_entry",
    model=model_name,
    description="Entry point for the summarization agent.",
    instruction="""
    - Ask the user for text to summarize.
    - When the user provides input, use 'add_prompt_to_state' to save it.
    - Then pass control to the summarization_workflow.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[summarization_workflow]
)
