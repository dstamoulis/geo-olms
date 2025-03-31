
from agent_core.agents.assistant_agent import AssistantAgent
from typing import Callable

from agent_core.modules.toolset import agent_toolset
from agent_core.modules.tool_schema import function_to_tool_json


class PseudoSolverAgent(AssistantAgent):
    def __init__(self, name, model_client, messages, tools=[], handoffs=None, system_message="", console=None):
        """
        PseudoSolverAgent behaves like an AssistantAgent that executes the low-code tools 'scripted' explicitly by the user
        As a wrapper of AssistantAgent, PseudoSolverAgent resets the agent's tools at each query.

        Args:
            name (str): The agent's name.
            model_client: An instance of an LLM client.
            messages: A Messages instance holding the conversation history.
            tools (list): Initially empty list (it is reset at every execution!).
            handoffs (list, optional): List of agent names for possible handoffs.
            system_message (str, optional): A system instruction message.
            console (optional): A reference to a higher-level orchestrator.
        """
        super().__init__(name, model_client, messages, handoffs, tools, system_message, console)

    def run_query(self, query, tool_choices):
        """
        NOTE: This emulates the forced-function tool_choice from OpenAI:
        https://platform.openai.com/docs/guides/function-calling#tool-choice
        While I could directly pass this as argument to the LLM client, it will only
        work for OpenAI and break things for other clients. E.g., Ollama does not support
        tool_choice argument yet!!!
        https://github.com/ollama/ollama/blob/main/docs/openai.md
        """

        # Ensure tool_choices is a list of callables.
        if not isinstance(tool_choices, list) or not all(callable(tool) for tool in tool_choices):
            return "Error! Running the pseudo-solver requires a list of callables"

        # Build tool_schemas and tools_map based on the new tools.
        self.tools = tool_choices
        self.tool_schemas = [function_to_tool_json(tool) for tool in self.tools]
        self.tools_map = {tool.__name__: tool for tool in self.tools}

        # Call the parent's run_query method.
        return super().run_query(query)

