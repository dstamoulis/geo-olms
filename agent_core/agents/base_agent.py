# agent_core/agents/base_agent.py

from agent_core.modules.messages import TextMessage

class BaseAgent:
    def __init__(self, name, model_client, messages, handoffs=None, tools=None, system_message=None, console=None):
        """
        Minimal BaseAgent which holds common parameters.
        
        Args:
            name (str): The agent's name.
            model_client: An LLM client instance.
            handoffs (list, optional): List of agent names for possible handoffs.
            tools (list, optional): List of tool functions available.
            system_message (str, optional): Instructional system message.
            console (optional): Reference to a higher-level orchestrator.
        """
        self.name = name
        self.model_client = model_client
        self.messages = messages
        self.handoffs = handoffs if handoffs is not None else []
        self.tools = tools if tools is not None else []
        self.system_message = system_message
        self.console = console

    # not used (because in Swarm the system top message changes based on the round!)
    # https://cookbook.openai.com/examples/orchestrating_agents
    def init_messages(self):
        if self.system_message is not None:
            text_message = TextMessage(role='system', content=self.system_message)
            self.messages.add_message(text_message)

    def log(self, message):
        print(f"[{self.name}] {message}")