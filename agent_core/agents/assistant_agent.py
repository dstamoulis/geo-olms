# agent_core/agents/assistant_agent.py

import time
import json
import re

from agent_core.agents.base_agent import BaseAgent
from agent_core.modules.messages import ChatResponseMessage, TextMessage, ToolCall, ToolCallRequestMessage, ToolResponseMessage
from agent_core.modules.tool_schema import function_to_tool_json, function_to_tool_json_Response

class AssistantAgent(BaseAgent):
    def __init__(self, name, model_client, messages, api="ChatCompletion", handoffs=[], tools=None, system_message="", console=None):
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
        # turn = 0
        num_init_messages = len(self.messages)

        while True:

            self.log(f"Requesting response from {self.model_client.client_class} client ({self.model_client.model})...")
            messages = [{"role": "system", "content": self.system_message}] if self.system_message is not None else []

            # experiment: return function call result without having a function call request
            # if turn == 1:
            #     print(f"#####\nmessages before tampering: {messages}\n#####")
            #     messages.append({
            #         'arguments': '{"lat":51.5074,"lon":-0.1278,"zoom":10}',
            #         'call_id': 'call_fake_1',
            #         'name': 'zoom_map',
            #         'type': 'function_call',
            #         'id': 'fc_fake_1',
            #         'status': 'completed'
            #     })
            #     messages.append({
            #         'type': 'function_call_output',
            #         'call_id': 'fake_call_1',
            #         'output': 'Zoom updated.'
            #     })
            #     print(f"#####\nmessages after tampering: {messages}\n#####")

            messages = messages + self.messages.get_client_messages(self.model_client.client_class)
            self.log(f"=====\nMessages sent to model: {messages}\n=====")
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

            # turn += 1

        return chat_response
    

# ------------------------------------------------
# Workflow execution
# ------------------------------------------------
    def get_response_workflow(self, task_id, workflow):
        while True:
            self.log(f"Requesting response from {self.model_client.client_class} client ({self.model_client.model})...")
            messages = [{"role": "system", "content": self.system_message}] if self.system_message is not None else []

            messages = messages + self.messages.get_client_messages(self.model_client.client_class)
            # self.log(f"Messages sent to model\n{messages}\n")
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

        workflow[task_id]['history'] = chat_response.content
        return chat_response

    def run_workflow(self, agents: dict, workflow: dict, ui_mode=False):
        agent_calls = ""
        for task_id, task in workflow.items():
            # print(f'task_id: {task_id}, task: {task}')
            agent_calls += f"{task_id}: {task["objective"]}, agent: {task["agent"]}\n"
        agent_calls += f"Here is the list of available agents: {list(agents.keys())}. For each task, match it with an agent that is strictly in the list, make a guess if you need but make sure it's in the list. Then just return only the one-to-one result in this format: task_num: agent_name"
        # print(f"************************ agent_calls: \n{agent_calls}")
        message = [{"role": "system", "content": agent_calls}]
        agent_match = self.model_client.get_response(message)
        print(f"---------------------Agent match response: \n{agent_match}")
        # agent_match = "\n".join(agent_match.content.strip().splitlines()[:-1])

        print(f"---------------------Here are the agents we have: {agents.keys()}, Here's the agent_match: {agent_match}")

        # Convert the LLM response into a dictionary of form: task_id: agent_name
        pattern = r"\s*(task\d+):\s*([a-zA-Z0-9_]+)" 
        matches = re.findall(pattern, agent_match.content)
        tool_calls = agent_match.tool_calls
        print(f'+++++++++++++++++++++++tool calls are: {agent_match}')
        task_agent_map = dict(matches)
        print(f"++++++++++++++++ map: \n{task_agent_map}")

        for task_id, task in workflow.items():
            # TODO: if completed, fetch history from ground truth
            if task['status'] == 'completed':
                print(f"Task {task_id} already completed. Skipping.")
                continue
            else:
                print(f"\nProcessing task {task_id} with objective: {task['objective']}")
                handoff_agent = agents[task_agent_map[task_id]]
                content = task['objective']
                text_message = TextMessage(role='user', content=content, source='user')
                handoff_agent.messages.add_message(text_message)
                response = handoff_agent.get_response_workflow(task_id, workflow)
        with open("target.json", "w") as f:
            json.dump(workflow, f, indent=2)
        return response.content if ui_mode else response
    
def get_context(task_id: str, workflow: dict):
    context_lines = []
    for prev_id in workflow[task_id]['prev']:
        if prev_id in workflow:
            context_lines.append(
                    f"Task {prev_id}:\n"
                    f"  Objective: {workflow[prev_id]['objective']}\n"
                    f"  Result: {workflow[prev_id]['history']}\n"
                )
        
        if context_lines:
            return "\n".join(context_lines)
        else:
            return "No completed previous tasks context available."
    return context_lines

def get_downstream_objectives(task_id: str, workflow: dict):
    downstream_objectives = []
    for next_id in workflow[task_id]['next']:
        if next_id in workflow:
            downstream_objectives.append(workflow[next_id]['objective'])
    return downstream_objectives if downstream_objectives else ["No downstream objectives available."]

def format_content(objective, context, downstream_objectives):
    return f"\n**Objective**\n{objective}\n\
            **Context from upstream tasks**\n{context if context else "No context available."}\n\
            **Downstream objectives**\n{downstream_objectives if downstream_objectives else 'No downstream objectives available.'}\n"