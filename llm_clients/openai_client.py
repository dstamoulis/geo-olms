# llm_clients/openai_client.py

import time
from openai import OpenAI
from llm_clients.base_client import BaseClient
from agent_core.modules.messages import ChatResponseMessage, ToolCall, ToolCallRequestMessage

class OpenAIClient(BaseClient):
    client_class: str = "OpenAIClient"

    def __init__(self, model, temperature=0.1, api_key=None, **kwargs):
        """
        Initialize the OpenAI client.

        Args:
            model (str): The model name (e.g., "gpt-4o-mini").
            api_key (str, optional): Your OpenAI API key.
            **kwargs: Additional parameters.
        """
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key
        self.api_client = OpenAI()

    # Chat Completion API
    def get_response(self, messages, tools=None):
        """
        Send the conversation history (and tools, if provided) to 
        OpenAI's API and return the response.

        Args:
            messages (list): Conversation history.
            tools (list, optional): Tool specifications.

        Returns:
            dict: A dictionary with keys like "response", "prompt_tokens", "completion_tokens".
        """
        start_time = time.time()
        response = self.api_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            # temperature=self.temperature,
        )
        elapsed_time = round(time.time() - start_time, 4)
        if tools is not None and response.choices[0].message.tool_calls:
            message_type = ToolCallRequestMessage
            response_tool_calls = []
            for tool in response.choices[0].message.tool_calls:
                tool_call = ToolCall(
                    id=tool.id,
                    name=tool.function.name,
                    arguments=tool.function.arguments,
                    tool_type=tool.type
                )
                response_tool_calls.append(tool_call)
        else:
            message_type = ChatResponseMessage
            response_tool_calls = None

        response_message = message_type(
            role='assistant', 
            content=response.choices[0].message.content,
            message=response.choices[0].message,
            prompt_tokens=response.usage.prompt_tokens,
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            time_elapsed=elapsed_time,
            tool_calls=response_tool_calls,
            source=self.client_info()
        )
        return response_message

    # Responses API
    def get_response_Response(self, messages, tools=None):
        """
        Send the conversation history (and tools, if provided) to 
        OpenAI's API and return the response.

        Args:
            messages (list): Conversation history.
            tools (list, optional): Tool specifications.

        Returns:
            dict: A dictionary with keys like "response", "prompt_tokens", "completion_tokens".
        """
        start_time = time.time()
        response = self.api_client.responses.create(
            model=self.model,
            input=messages,
            tools=tools,
            temperature=self.temperature,
        )
        elapsed_time = round(time.time() - start_time, 4)
        response_content = None
        response_tool_calls = None

        message_type = ChatResponseMessage
        for item in response.output:
            # Handle function calls
            if item.type == "function_call":
                message_type = ToolCallRequestMessage
                tool_call = ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                    tool_type=item.type
                )
                if response_tool_calls is None:
                    response_tool_calls = []
                response_tool_calls.append(tool_call)
            # Handle messages
            elif item.type == "message":
                # Assumes the content field contains only one message
                if response_content is None:
                    response_content = item.content[0].text
                else:
                    response_content += item.content[0].text

        response_message = message_type(
            role='assistant',
            content=response_content,
            message=[res.model_dump() for res in response.output],
            prompt_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            time_elapsed=elapsed_time,
            tool_calls=response_tool_calls,
            source=self.client_info()
        )
        return response_message
    

    def to_dict(self):
        return {
            'client_class': self.client_class,
            'model': self.model,
            'temperature': self.temperature
        }