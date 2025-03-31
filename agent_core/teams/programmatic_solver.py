import json
from typing import Callable

from agent_core.agents.assistant_agent import AssistantAgent
from agent_core.modules.messages import UserProgramToolCallRequest, ToolCall

from agent_core.modules.toolset import agent_toolset
from agent_core.modules.tool_schema import function_to_tool_json

class ProgrammaticSolver(AssistantAgent):
    def __init__(self, name, messages, toolsets_list):
        """
        ProgrammaticSolver uses the Agent-based message and tool-execution logic WITHOUT calling
        the LLM endpoint. Instead it executes the tool-call as explicitly defined by the user!

        Args:
            name (str): The agent's name.
            messages: A Messages instance holding the conversation history.
            tools (list): Initially empty list (it is reset at every execution!).
        """
        # Aggregate all tools from the provided toolset objects.
        aggregated_tools = []
        for toolset in toolsets_list:
            # extract_agent_toolset is a utility function that returns a dict of {tool_name: callable}
            toolset_list = agent_toolset(toolset)
            # We add the tool callables to our aggregated list.
            aggregated_tools.extend(toolset_list)

        handoffs = console = model_client = None
        system_message = ""
        super().__init__(name, model_client, messages, handoffs, aggregated_tools, system_message, console)
        self.toolsets_list = toolsets_list

    def exec_tool(self, tool, tool_args):

        tool_calls = [ToolCall(
            id=0,
            name=tool.__name__,
            arguments=json.dumps(tool_args),
            tool_type="function"
        )]
    
        user_tool_call_msg = UserProgramToolCallRequest(
            role='programmatic', 
            tool_calls=tool_calls,
            source='programmatic'
        )
        self.messages.add_message(user_tool_call_msg)

        tool_response = super().execute_tool_call(tool_calls[0])
        self.messages.add_message(tool_response)

        return "Tool executed"

