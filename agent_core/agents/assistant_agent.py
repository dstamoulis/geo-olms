# agent_core/agents/assistant_agent.py

import time
import json
import re

from agent_core.agents.base_agent import BaseAgent
from agent_core.modules.messages import ChatResponseMessage, TextMessage, ToolCall, ToolCallRequestMessage, ToolResponseMessage
from agent_core.modules.tool_schema import function_to_tool_json, function_to_tool_json_Response
from utils import extract_json_object, parse_agent_decision

from collections import deque

def ordered_workflow_tasks(workflow: dict) -> list[str]:
    """
    Given a dict whose keys are task IDs like "task0", "task1", ..., "task10", 
    return a list of those keys sorted by their numeric suffix:
      ["task0", "task1", "task2", ..., "task10", "task11", ...]
    """
    def sort_key(task_id: str):
        # extract the trailing number (e.g. "10" from "task10")
        m = re.search(r'(\d+)$', task_id)
        return int(m.group(1)) if m else float('inf')

    return sorted(workflow.keys(), key=sort_key)


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
        while True:

            self.log(f"Requesting response from {self.model_client.client_class} client ({self.model_client.model})...")
            messages = [{"role": "system", "content": self.system_message}] if self.system_message is not None else []

            messages = messages + self.messages.get_client_messages(self.model_client.client_class)
            self.log(f"Messages sent to model\n{messages}\n")
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
                if self.api == "Responses":
                    tool_response = self.execute_tool_call_Response(tool_call)
                else:
                    tool_response = self.execute_tool_call(tool_call)
                self.messages.add_message(tool_response)

        return chat_response
    
    def get_response_subagent(self, orch_agent):
        """
        Retrieves the conversation history in API format, then calls the model client to get a response.
        """
        while True:

            self.log(f"Requesting response from {self.model_client.client_class} client ({self.model_client.model})...")
            messages = [{"role": "system", "content": self.system_message}] if self.system_message is not None else []

            messages = messages + self.messages.get_client_messages(self.model_client.client_class)
            self.log(f"Messages sent to model\n{messages}\n")
            if self.api == "Responses":
                chat_response = self.model_client.get_response_Response(messages, tools=self.tool_schemas)
            else:
                chat_response = self.model_client.get_response(messages, tools=self.tool_schemas)
            self.log(f"Received response: {chat_response}")
            self.messages.add_message(chat_response)
            orch_agent.messages.add_message(chat_response)

            if not isinstance(chat_response, ToolCallRequestMessage):  # if finished handling tool calls, break
                break

            # === handle tool calls ===
            for tool_call in chat_response.tool_calls:
                if self.api == "Responses":
                    tool_response = self.execute_tool_call_Response(tool_call)
                else:
                    tool_response = self.execute_tool_call(tool_call)
                self.messages.add_message(tool_response)
                orch_agent.messages.add_message(tool_response)

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
                if self.api == "Responses":
                    tool_response = self.execute_tool_call_Response(tool_call)
                else:
                    tool_response = self.execute_tool_call(tool_call)
                self.messages.add_message(tool_response)

        return chat_response
    
    # ------------------------------------------------------------------------------
    # Main loop for executing GeoFlow
    def run_workflow(self, agents: dict, workflow: dict, ui_mode=False, log_target=False):

        # 1) Build ordered list of objectives
        ordered_task_ids = ordered_workflow_tasks(workflow)

        # 2) Execute workflow
        for task_id in ordered_task_ids:
            task = workflow[task_id]
            print(f"\nProcessing task {task_id} with objective: {task['objective']}")
            handoff_agent = agents[task['agent']]
            text_message = TextMessage(role='user', content=task['objective'], source='user')
            print(f"\nText message: {text_message}\n")
            handoff_agent.messages.add_message(text_message)
            response = handoff_agent.get_response()
            print(f"\nResponse: {response}\n")
        return response.content if ui_mode else response
    
    # ------------------------------------------------------------------------------
    # Updated loop for executing "geoolm" (Dimi!)
    def run_workflow_flow(self, agents: dict, workflow: dict, ui_mode=False, log_target=False):
        """
        1. Extract objectives in ID order.
        2. Prompt the LLM to assign each task to one of the given agents.
        3. Retry up to 3 times if the JSON is invalid or contains unknown agents.
        4. Return the assignment dict.
        """
        # 1) Build ordered list of objectives
        ordered_task_ids = ordered_workflow_tasks(workflow)
        agent_task_objectives = {tid: workflow[tid]["objective"] for tid in ordered_task_ids}
        
        # 2) Build the prompt
        llm_prompt = f"""
            For each of the following tasks and their objectives, assign exactly one agent.
            You may choose only from: {list(agents.keys())}.

            Each task is self-contained and maps to a single agent:

            > database_agent: tools for database queries and image filtering  
            > detector_agent: detection and classification on prior images  
            > data_agent: data-count and analytics tools  
            > map_agent: map zooming, plotting, and visualization tools  

            Reply ONLY with a JSON object mapping task IDs to agent names.  
            DO NOT include any backticks, commentary, or extra text—only the raw JSON.

            Example valid output:
            {{"task0": "database_agent", "task1": "detector_agent", "task2": "map_agent"}}

            Here are the tasks to assign:
            {json.dumps(agent_task_objectives, indent=2)}
            """

        # 3) LLM call to assign agents to objectives
        max_retries = 3
        attempts = 0
        agent_assignment = None

        while attempts < max_retries:
            attempts += 1
            response = self.model_client.get_response([  # replace with your LLM call
                {"role": "system", "content": "You are an expert workflow allocator. Given a list of available agent names, your task is to assign the proper agent to each objective!"},
                {"role": "user",   "content": llm_prompt}
            ])
            raw = response.content
            try:
                # extract_json_object as defined earlier
                json_str = extract_json_object(raw)
                candidate_agent_assignment = json.loads(json_str)
                # 4) Validate agent names
                if all(agent in agents for agent in candidate_agent_assignment.values()):
                    agent_assignment = candidate_agent_assignment
                    break
            except Exception as e:
                print(f"[Attempt {attempts}/{max_retries}] Failed to produce agent-assignement JSON: {e}")
                pass

        if agent_assignment is None:
            print("Failed to get valid agent assignment after retries")
            return 

        print(f"[Attempt {attempts}/{max_retries}] Produced agent-assignement JSON: {agent_assignment}")

        # 4) Execute workflow
        for task_id in ordered_task_ids:
            task_objective = agent_task_objectives[task_id]
            handoff_agent = agents[agent_assignment[task_id]]
            print(f"\nProcessing task {task_id} with objective: {task_objective} and agent: {agent_assignment[task_id]}")
            text_message = TextMessage(role='user', content=task_objective, source='user')
            print(f"\nText message: {text_message}\n")
            handoff_agent.messages.add_message(text_message)
            response = handoff_agent.get_response()
            print(f"\nResponse: {response}\n")
        return response.content if ui_mode else response
    
    # ------------------------------------------------------------------------------
    # Main loop for executing Seq-StateFlow (Dimi)
    def run_seq_stateflow(self, query, handoff_message: str, agents: dict, agents_sequence: list, ui_mode=False, log_target=False):

        sequence_reset = True
        resets_cnt = 0
        while sequence_reset:

            sequence_reset = False
            for next_agent_name in agents_sequence:

                next_agent = agents[next_agent_name]
                text_message = TextMessage(role='user', content=query+handoff_message, source='user')
                next_agent.messages.add_message(text_message)
                response = next_agent.get_response()
                next_agent.messages.add_message(response)

                transition_decision = parse_agent_decision(response.content)
                if transition_decision == "ERROR":
                    resets_cnt+=1
                    sequence_reset = resets_cnt < 2 # so if you reset twice already, give up
                    break

        return response.content if ui_mode else response


    # ------------------------------------------------------------------------------
    # Main loop for executing Group StateFlow, with orch_message as system message
    def run_group_stateflow_sys(self, handoffs: dict, query: str, ui_mode=False):
        error_cnt = 0
        errorout = False
        text_message = TextMessage(role='user', content=query, source='user')
        self.messages.add_message(text_message)
        
        while True:
            if errorout:
                break
            
            orch_response = self.get_response()
            handoff_decision = parse_agent_decision(orch_response.content)

            if handoff_decision == "DONE":
                return orch_response.content if ui_mode else orch_response
            elif handoff_decision == "database_agent" or handoff_decision == "map_agent" or handoff_decision == "detector_agent" or handoff_decision == "data_agent":
                # hand off the task to the corresponding agent
                handoff_agent = handoffs[handoff_decision]
                print(f"\nHandoffing to {handoff_agent.name}...")
                handoff_agent.get_response()
            else:
                error_cnt += 1
                errorout = error_cnt < 2

        return orch_response.content if ui_mode else orch_response
    
    # ------------------------------------------------------------------------------
    # Main loop for executing GC StateFlow, with orch_message as user message
    def run_group_stateflow_usr(self, handoffs: dict, orch_message: str, query: str, ui_mode=False):
        error_cnt = 0
        errorout = False
        
        while True:
            if errorout:
                break
            
            text_message = TextMessage(role='user', content=f"*****\n{orch_message}\n*****\nNow please help me with the task: {query}", source='user')
            self.messages.add_message(text_message)

            orch_response = self.get_response()
            handoff_decision = parse_agent_decision(orch_response.content)
            
            if handoff_decision == "DONE":
                return orch_response.content if ui_mode else orch_response
            elif handoff_decision == "database_agent" or handoff_decision == "map_agent" or handoff_decision == "detector_agent" or handoff_decision == "data_agent":
                # hand off the task to the corresponding agent
                handoff_agent = handoffs[handoff_decision]
                print(f"Handing off to {handoff_agent.name}...")
                if not handoff_agent.messages:
                    handoff_message = TextMessage(role='user', content=query, source='user')
                    handoff_agent.messages.add_message(handoff_message)
                handoff_agent.get_response_subagent(self)
            else:
                error_cnt += 1
                errorout = error_cnt < 2

        return orch_response.content if ui_mode else orch_response
    
    # ------------------------------------------------------------------------------
    # Main loop for executing Flow++
    def run_flowPP(self, agents: dict, workflow: dict, ui_mode=False, log_target=False):
        agent_calls = ""
        for task_id, task in workflow.items():
            # print(f'task_id: {task_id}, task: {task}')
            agent_calls += f"{task_id}: {task['objective']}, agent: {task['agent']}\n"
        agent_calls += f"Here is the dict_keys of available agents: {agents.keys()}. For each task, match it with an agent that is strictly in the dict_keys, make a guess if you need. Then just return only the one-to-one result in this format: task_num: agent_name"
        print(f"************************ agent_calls: \n{agent_calls}")

        # agent_match = self.model_client.get_response_Response(agent_calls)
        message = [{"role": "system", "content": agent_calls}]
        agent_match = self.model_client.get_response(message)
        print(f"---------------------Agent match response: \n{agent_match.content}")
        tool_calls = agent_match.tool_calls

        # Convert the LLM response into a dictionary of form: task_id: agent_name
        pattern = r"\s*(task\d+):\s*([a-zA-Z0-9_]+)" 
        matches = re.findall(pattern, agent_match.content)
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

        if log_target:
            with open("target.json", "w") as f:
                json.dump(workflow, f, indent=2)
        return response.content if ui_mode else response