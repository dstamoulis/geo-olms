# agent_core/agents/assistant_agent.py

import time
import json

from agent_core.agents.base_agent import BaseAgent
from agent_core.modules.messages import ChatResponseMessage, TextMessage, ToolCall, ToolCallRequestMessage, ToolResponseMessage
from agent_core.modules.tool_schema import function_to_tool_json, function_to_tool_json_Response

class AssistantAgent(BaseAgent):
    def __init__(self, api, name, model_client, messages, handoffs=[], tools=None, system_message="", console=None):
        """
        AssistantAgent extends BaseAgent with tool calling and orchestration logic.
        """
        super().__init__(name, model_client, messages, handoffs, tools, system_message, console)
        self.api = api


        # turn python functions into tools and save a reverse map
        # following: https://cookbook.openai.com/examples/orchestrating_agents
        assert tools is not None and isinstance(tools, list)
        assert isinstance(handoffs, list)

        self.tools = tools + handoffs
        if api == "Responses":
            self.tool_schemas = [function_to_tool_json_Response(tool) for tool in self.tools]
        else:
            self.tool_schemas = [function_to_tool_json(tool) for tool in self.tools]
        self.tools_map = {tool.__name__: tool for tool in self.tools}
    
    def run_query(self, query, ui_mode=False):
        """
        Add the incoming message to the conversation history,
        then call get_response() to obtain a reply from the model.
        """
        self.log("Adding user message to conversation history.")
        text_message = TextMessage(role='user', content=query, source='user')
        self.messages.add_message(text_message)
        print(f"User message: {self.messages}")
        response = self.get_response()
        return response.content if ui_mode else response
    

    def execute_tool_call(self, tool_call):
        
        name = tool_call.name
        args = json.loads(tool_call.arguments)
        self.log(f"Calling tool: {name}({args})")

        # call corresponding function with provided arguments
        start_time = time.time()
        tool_response = self.tools_map[name](**args)
        elapsed_time = round(time.time() - start_time, 4)
        
        llm_as_a_tool_response = False
        result_message = {"role": "tool", "tool_call_id": tool_call.id}
        if isinstance(tool_response, int) or isinstance(tool_response, float):
            result_message["content"] = str(tool_response)
        elif isinstance(tool_response, str):
            result_message["content"] = tool_response
        elif isinstance(tool_response, ChatResponseMessage): # LLM-as-a-Tool!
            llm_as_a_tool_response = True
            result_message["content"] = tool_response.content
        else:
            raise ValueError(f"Unsupported function return type: Tool {name} | Args {args} | Return {type(tool_response)}")

        return ToolResponseMessage(
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
    
    def execute_tool_call_Response(self, tool_call):
        
        name = tool_call.name
        args = json.loads(tool_call.arguments)
        self.log(f"Calling tool: {name}({args})")

        # call corresponding function with provided arguments
        start_time = time.time()
        tool_response = self.tools_map[name](**args)
        elapsed_time = round(time.time() - start_time, 4)
        
        llm_as_a_tool_response = False
        result_message = {"type": "function_call_output", "call_id": tool_call.id}
        if isinstance(tool_response, int) or isinstance(tool_response, float):
            result_message["output"] = str(tool_response)
        elif isinstance(tool_response, str):
            result_message["output"] = tool_response
        elif isinstance(tool_response, ChatResponseMessage): # LLM-as-a-Tool!
            llm_as_a_tool_response = True
            result_message["output"] = tool_response.content
        else:
            raise ValueError(f"Unsupported function return type: Tool {name} | Args {args} | Return {type(tool_response)}")

        return ToolResponseMessage(
            role='function_call_output', 
            content=result_message["output"],
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
            if self.api == "Responses":
                chat_response = self.model_client.get_response_Response(messages, tools=self.tool_schemas)
            else:
                chat_response = self.model_client.get_response(messages, tools=self.tool_schemas)
            self.log(f"Received response: {chat_response}")
            self.messages.add_message(chat_response)

            if not isinstance(chat_response, ToolCallRequestMessage):  # if finished handling tool calls, break
                break

            # === handle tool calls ===
            for tool_call in chat_response.tool_calls:
                # tool_response = self.execute_tool_call(tool_call)
                if self.api == "Responses":
                    tool_response = self.execute_tool_call_Response(tool_call)
                else:
                    tool_response = self.execute_tool_call(tool_call)
                self.messages.add_message(tool_response)

        return chat_response

