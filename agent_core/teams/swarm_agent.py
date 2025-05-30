"""
Note (Jan 2025):
Swarm follows the educational implementation outlined in the OpenAI Cookbook example
for orchestrating agents: https://cookbook.openai.com/examples/orchestrating_agents.
AutoGen also provides an adaptation of Swarm but introduces heavily AutoGen-specific logic:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/swarm.html

Update (March 2025):
Swarm has been replaced by the OpenAI Agents SDK 
(https://github.com/openai/openai-agents-python), which now provides native API support for 
Swarm-based agentic logic (https://openai.github.io/openai-agents-python/agents/).

Roadmap:
Migrate to the OpenAI Agents API.
"""

import time
import json

from agent_core.agents.assistant_agent import AssistantAgent
from agent_core.modules.toolset import agent_toolset
from agent_core.agents.base_agent import BaseAgent
from agent_core.modules.messages import ChatResponseMessage, TextMessage, ToolCall, ToolCallRequestMessage, ToolResponseMessage, HandoffMessage
from agent_core.modules.tool_schema import function_to_tool_json


class SwarmAgent:
    def __init__(
        self, 
        name, 
        model_client, 
        messages, 
        database, 
        detector, 
        map_ops,
        triage,
        system_message="", 
        ):
        """
        SwarmAgent (re)implements BaseAgent with Swarm-based orchestration (handoffs) 
        logic on top of the tool-calling flow.
        """
        self.name = name
        self.model_client = model_client
        self.messages = messages
        self.handoffs = []
        self.tools = []
        self.system_message = system_message
        self.console = None # legacy

        self.current_agent = None
        self.new_agent = None

        self.workflow = [] # that's the json-based workflow definition!

        self.detector = detector
        self.database = database
        self.map_ops = map_ops
        self.triage = triage

        self.triage_agent = AssistantAgent(
            name="triage_agent",
            model_client=model_client,
            messages=messages,
            handoffs=agent_toolset(triage),
            tools=[],
            system_message="Pass the request off to a SINGLE ONE AND ONLY HANDOFF AGENT!! Do NOT call multiple handoffs at once!!"
            )
        

        self.database_agent = AssistantAgent(
            name="database_agent",
            model_client=model_client,
            messages=messages,
            handoffs=agent_toolset(triage),
            tools=agent_toolset(database),
            system_message="You are the database agent!"
            )

        self.map_agent = AssistantAgent(
            name="map_agent",
            model_client=model_client,
            messages=messages,
            handoffs=agent_toolset(triage),
            tools=agent_toolset(map_ops),
            system_message="You are the map agent!"
            )

        self.detector_agent = AssistantAgent(
            name="detector_agent",
            model_client=model_client,
            messages=messages,
            handoffs=agent_toolset(triage),
            tools=agent_toolset(detector),
            system_message="You are the detector agent!"
            )

        self.triage.database_agent = self.database_agent
        self.triage.map_agent = self.map_agent
        self.triage.detector_agent = self.detector_agent

        self.new_agent = self.database_agent
        self.update_current_agent()

    
    def run_query(self, query, ui_mode=False):
        """
        Add the incoming message to the conversation history,
        then call get_response() to obtain a reply from the model.
        """
        self.log("Adding user message to conversation history.")
        text_message = TextMessage(role='user', content=query, source='user')
        self.messages.add_message(text_message)
        response = self.get_response()
        return response.content if ui_mode else response


    def update_current_agent(self):
        self.current_agent = self.new_agent
        self.tool_schemas = self.new_agent.tool_schemas
        self.tools_map = self.new_agent.tools_map


    def execute_tool_call(self, tool_call):
        
        name = tool_call.name
        args = json.loads(tool_call.arguments)
        self.log(f"Calling tool: {name}({args})")

        # call corresponding function with provided arguments
        start_time = time.time()
        tool_response = self.tools_map[name](**args)
        elapsed_time = round(time.time() - start_time, 4)
        
        llm_as_a_tool_response = False
        handoff_response = False

        message_type = ToolResponseMessage
        result_message = {"role": "tool", "tool_call_id": tool_call.id}
        if isinstance(tool_response, int) or isinstance(tool_response, float):
            result_message["content"] = str(tool_response)
        elif isinstance(tool_response, str):
            result_message["content"] = tool_response
        elif isinstance(tool_response, ChatResponseMessage): # LLM-as-a-Tool!
            llm_as_a_tool_response = True
            result_message["content"] = tool_response.content
        elif isinstance(tool_response, AssistantAgent) or isinstance(tool_response, BaseAgent): # HandOff!
            message_type = HandoffMessage
            self.new_agent = tool_response
            result_message["content"] = f"Transfered to {self.new_agent.name}. Adopt persona immediately"
        else:
            raise ValueError(f"Unsupported function return type: Tool {name} | Args {args} | Return {type(tool_response)}")

        return message_type(
            role='tool', 
            content=result_message["content"],
            message=result_message,
            prompt_tokens=None if not llm_as_a_tool_response else tool_response.prompt_tokens,
            cached_tokens=None if not llm_as_a_tool_response else tool_response.cached_tokens,
            completion_tokens=None if not llm_as_a_tool_response else tool_response.completion_tokens,
            total_tokens=None if not llm_as_a_tool_response else tool_response.total_tokens,
            time_elapsed=elapsed_time,
            source=tool_call.name
        )

    def get_response(self):
        """
        Retrieves the conversation history in API format, then calls the model client to get a response.
        """
        num_init_messages = len(self.messages)

        while True:

            self.log(f"Requesting response from {self.model_client.client_class} client ({self.model_client.model})...")
            messages = [{"role": "system", "content": self.system_message}] if self.system_message is not None else []
            messages = messages + self.messages.get_client_messages(self.model_client.client_class)
            chat_response = self.model_client.get_response(messages, tools=self.tool_schemas)
            self.log(f"Received response: {chat_response}")
            self.messages.add_message(chat_response)

            if not isinstance(chat_response, ToolCallRequestMessage):  # if finished handling tool calls, break
                break

            # === handle tool calls ===
            for tool_call in chat_response.tool_calls:
                tool_response = self.execute_tool_call(tool_call)
                self.messages.add_message(tool_response)

                # following -- https://cookbook.openai.com/examples/orchestrating_agents#handoff-functions -- exactly!
                if isinstance(tool_response, HandoffMessage):
                    self.update_current_agent()
                    # break


        return chat_response


    def log(self, message):
        print(f"[{self.name}] {message}")
