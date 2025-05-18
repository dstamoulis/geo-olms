
from agent_core.agents.assistant_agent import AssistantAgent
from agent_core.modules.toolset import agent_toolset

class SingleAgent(AssistantAgent):
    def __init__(self, api, name, model_client, messages, toolsets_list: list, handoffs=[], system_message="", console=None):
        """
        SingleAgent aggregates tools from multiple toolset objects and behaves like an AssistantAgent.
        
        Args:
            name (str): The agent's name.
            model_client: An instance of an LLM client.
            messages: A Messages instance holding the conversation history.
            toolsets_list (list): A list of objects (e.g., memory, map_ops, detection_obj) from which agent tools are extracted.
            handoffs (list, optional): List of agent names for possible handoffs.
            system_message (str, optional): A system instruction message.
            console (optional): A reference to a higher-level orchestrator.
        """
        # Aggregate all tools from the provided toolset objects.
        aggregated_tools = []
        for toolset in toolsets_list:
            # extract_agent_toolset is a utility function that returns a dict of {tool_name: callable}
            toolset_list = agent_toolset(toolset)
            # We add the tool callables to our aggregated list.
            aggregated_tools.extend(toolset_list)
        
        super().__init__(api, name, model_client, messages, handoffs, aggregated_tools, system_message, console)
        self.toolsets_list = toolsets_list