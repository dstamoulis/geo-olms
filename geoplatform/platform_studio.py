# geoplatform/platform.py

from agent_core.teams.single_agent import SingleAgent

class PlatformStudio:
    def __init__(self, model_client, messages, database, vision, map_tools, data_tools):
        """
        Initialize the PlatformStudio with all system components where each API is an agent!

        Args:
            model_client: Instance of your LLM client.
            messages (Messages): The conversation/messages container.
            database: Instance of Database (or similar) for memory management.
            vision: Instance of your Vision object.
            map_tools: Instance of your Map operations object.
            data_tools: Instance of your Data operations object.
        """
        self.model_client = model_client
        self.messages = messages
        self.database = database
        self.vision = vision
        self.map_tools = map_tools

        self.database_agent = SingleAgent(
            name="database_agent",
            model_client=model_client,
            messages=messages,
            toolsets_list=[database],
            system_message="You are a geospatial agent helping with fetching images from a database!"
        )

        self.map_agent = SingleAgent(
            name="map_agent",
            model_client=model_client,
            messages=messages,
            toolsets_list=[map_tools],
            system_message="You are a geospatial agent helping with fetching images from a database!"
        )

        self.vision_agent = SingleAgent(
            name="vision_agent",
            model_client=model_client,
            messages=messages,
            toolsets_list=[vision],
            system_message="You are a geospatial agent helping with fetching images from a database!"
        )

        self.data_agent = SingleAgent(
            name="data_agent",
            model_client=model_client,
            messages=messages,
            toolsets_list=[data_tools],
            system_message="You are a geospatial agent helping with fetching images from a database!"
        )

        self.agents_mapping = {}
        self.agents_mapping["DatabaseAgent"] = self.database_agent
        self.agents_mapping["MapAgent"] = self.map_agent
        self.agents_mapping["VisionAgent"] = self.vision_agent
        self.agents_mapping["DataAgent"] = self.data_agent


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

    def run_workflow(self, workflow_dict):
        """
        Method to run a workflow parsing a provided .json.
        """

        query = workflow_dict[0].get("user query", None)
        if query is None: return "Invalid query"

        workflow_agents = workflow_dict[-1].get("config", {}).get("participants", {})
        if not workflow_agents: return "Invalid workflow"

        # print(workflow_agents)

        for workflow_agent in workflow_agents:
            agent_label = workflow_agent.get("label", "")
            if agent_label not in self.agents_mapping: continue
            self.agents_mapping[agent_label].run_query(query)

        response = "Done!"
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
