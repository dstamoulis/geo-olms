# geoplatform/platform.py

class Platform:
    def __init__(self, model_client, messages, database, vision, map_tools, agent):
        """
        Initialize the Platform with all system components.

        Args:
            model_client: Instance of your LLM client.
            messages (Messages): The conversation/messages container.
            database: Instance of Database (or similar) for memory management.
            vision: Instance of your Vision object.
            map_tools: Instance of your Map operations object.
            agent: The agent instance (e.g., FlatAgent) that orchestrates the interaction.
        """
        self.model_client = model_client
        self.messages = messages
        self.database = database
        self.vision = vision
        self.map_tools = map_tools
        self.agent = agent

    def reset(self):
        """
        Resets the state of the Platform by resetting the agent, messages, and any other stateful components.
        Adjust or extend the reset calls as necessary for your implementation.
        """
        self.messages.reset_messages()
        # self.agent.reset_agent()
        self.database.reset_database()
        self.vision.reset_vision()
        self.map_tools.reset_map()

    def run_query(self, query):
        """
        Convenience method to run a query through the agent.
        This may also include any pre- or post-processing specific to the platform.
        """
        # Optionally, you could include logging or pre-run logic here.
        response = self.agent.run_query(query)
        return response

    def get_state(self):
        """
        Optionally expose the current state (e.g., conversation history).
        """
        return {
            "messages": self.messages.to_list_dict(),
            "database": self.database.__dict__ if hasattr(self.database, "__dict__") else str(self.database),
            "map": self.map_tools.__dict__ if hasattr(self.map_tools, "__dict__") else str(self.map_tools)
            # Add other components as needed.
        }
