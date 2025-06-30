# agent_core/agents/assistant_agent.py

import time
import json
import re

from agent_core.agents.base_agent import BaseAgent
from agent_core.modules.messages import ChatResponseMessage, TextMessage, ToolCall, ToolCallRequestMessage, ToolResponseMessage
from agent_core.modules.tool_schema import function_to_tool_json, function_to_tool_json_Response

from collections import deque, OrderedDict
from utils import extract_json_object, parse_llm_delimiter_message

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
    # Updated loop for executing "geoolm"
    def run_workflow_flow(self, agents: dict, workflow: dict, orch_instructions: str, ui_mode=False, log_target=False):
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
                {"role": "user",   "content": orch_instructions + llm_prompt}
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
    # Main loop for executing Seq-StateFlow
    def run_seq_stateflow(self, query, subtask_instructions: str, agents: dict, agents_sequence: list, subagents_instructions: list, ui_mode=False):

        sequence_reset = True
        max_sequence_resets = 2
        resets_cnt = 0
        while sequence_reset:

            sequence_reset = False
            for next_agent_name in agents_sequence:

                next_agent = agents[next_agent_name]
                subtask_example = subagents_instructions[next_agent_name]["example"] if subagents_instructions else ""
                subtask_scope = subagents_instructions[next_agent_name]["scope"] if subagents_instructions else ""
                llm_promt = subtask_instructions.format(next_agent_name=next_agent_name, scope=subtask_scope, subtask_example=subtask_example)
                text_message = TextMessage(role='user', content=llm_promt + query, source='user')
                # print(f"\nText message: {text_message}\n")
                next_agent.messages.add_message(text_message)
                response = next_agent.get_response()
                print(f"\nResponse: {response}\n")

                if parse_llm_delimiter_message(response.content, "ERROR"):
                    resets_cnt+=1
                    sequence_reset = resets_cnt < max_sequence_resets # so if you reset twice already, give up
                    break

        return response.content if ui_mode else response

        
    # ------------------------------------------------------------------------------
    # Main loop for executing Group StateFlow
    def run_group_manager(self, query: str, agents: dict, orch_instructions: str, subtask_instructions: str, subagents_instructions: dict, ui_mode=False):
        
        max_orchestrator_steps = 10  # large-value to unstuck inf-loops
        orchestrator_steps = 0
        
        while orchestrator_steps <= max_orchestrator_steps:
            orchestrator_steps +=1

            orchestrator_llm_prompt = orch_instructions + query
            text_message = TextMessage(role='user', content=orchestrator_llm_prompt, source='user')
            self.messages.add_message(text_message)
            
            orch_response = self.get_response()

            next_agent_name = None
            raw = orch_response.content
            try:
                # extract_json_object as defined earlier
                json_str = extract_json_object(raw)
                candidate_agent_assignment = json.loads(json_str)
                _, next_agent_name = next(iter(candidate_agent_assignment.items()))

            except Exception as e:
                print(f"Attempt failed to produce agent-assignement JSON: {e}")
                pass

            if next_agent_name is not None and next_agent_name in agents:

                # hand off the task to the corresponding agent
                handoff_agent = agents[next_agent_name]
                print(f"\nHandoffing to {handoff_agent.name}...")
                subtask_example = subagents_instructions[next_agent_name]["example"] if subagents_instructions else ""
                subtask_scope = subagents_instructions[next_agent_name]["scope"] if subagents_instructions else ""
                subagents_llm_prompt = subtask_instructions.format(next_agent_name=next_agent_name, scope=subtask_scope, subtask_example=subtask_example)
                text_message = TextMessage(role='user', content=subagents_llm_prompt + query, source='user')
                handoff_agent.messages.add_message(text_message)
                response = handoff_agent.get_response()
                print(f"\nResponse: {response}\n")

            elif next_agent_name == "end_agent":

                return orch_response.content if ui_mode else orch_response       

        return orch_response.content if ui_mode else orch_response
    

    # ------------------------------------------------------------------------------
    # Main loop for executing Group StateFlow
    def run_magentic_stateflow(self, query: str, agents: dict, orch_instructions: str, subagents_instructions: dict, ui_mode=False):
        
        max_orchestrator_steps = 10 # large-value to unstuck inf-loops
        orchestrator_steps = 0
        
        while orchestrator_steps <= max_orchestrator_steps:
            orchestrator_steps +=1

            llm_prompt = f"""
                {orch_instructions}

                Now solve and continue with the following user task:

                {query}
                """

            text_message = TextMessage(role='user', content=llm_prompt, source='user')
            self.messages.add_message(text_message)
            orch_response = self.get_response()

            handoff_decision = None
            raw = orch_response.content
            try:
                # extract_json_object as defined earlier
                json_str = extract_json_object(raw)
                candidate_agent_assignment = json.loads(json_str)
                handoff_decision, handoff_objective = next(iter(candidate_agent_assignment.items()))

            except Exception as e:
                print(f"Attempt failed to produce agent-assignement JSON: {e}")
                pass

            if handoff_decision is not None and handoff_decision in agents:
                # hand off the task to the corresponding agent
                handoff_agent = agents[handoff_decision]
                handoff_prefix = subagents_instructions[handoff_decision] if subagents_instructions else ""
                print(f"\nHandoffing to {handoff_agent.name}...")
                text_message = TextMessage(role='user', content=handoff_prefix+handoff_objective, source='user')
                handoff_agent.messages.add_message(text_message)
                response = handoff_agent.get_response()
                print(f"\nResponse: {response}\n")

            elif handoff_decision == "end_agent":

                return orch_response.content if ui_mode else orch_response                

        return orch_response.content if ui_mode else orch_response
    
