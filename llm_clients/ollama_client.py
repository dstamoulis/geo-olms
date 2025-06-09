# llm_clients/ollama_client.py

import time, json
from ollama import ChatResponse, chat
from llm_clients.base_client import BaseClient
from agent_core.modules.messages import ChatResponseMessage, ToolCall, ToolCallRequestMessage


class OllamaClient(BaseClient):
    client_class: str = "OllamaClient"

    def __init__(self, model, temperature=0.1, num_predict=-1, **kwargs):
        """
        Initialize the Ollama client.

        Args:
            model (str): The model name.
            **kwargs: Additional parameters.
        """
        super().__init__(model, temperature, **kwargs)
        self.num_predict = num_predict
        self.num_ctx = self.get_num_ctx(model) 

    def get_response(self, messages, tools=None):
        """
        Send the conversation history (and tools, if provided) to 
        Ollama's API (running locally) and return the response.

        Args:
            messages (list): Conversation history.
            tools (list, optional): Tool specifications.

        Returns:
            dict: A dictionary with keys like "response", "prompt_tokens", "completion_tokens".
        """
        start_time = time.time()
        response: ChatResponse = chat(
            self.model,
            messages=messages,
            tools=tools,
            options= {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict
            }
        )
        elapsed_time = round(time.time() - start_time, 4)
        if response.message.tool_calls:
            message_type = ToolCallRequestMessage
            response_tool_calls = []
            for tool in response.message.tool_calls:
                tool_call = ToolCall(
                    id=getattr(tool, 'id', 0),
                    name=tool.function.name,
                    arguments=json.dumps(tool.function.arguments),
                    tool_type=getattr(tool, 'type', "function")
                )
                response_tool_calls.append(tool_call)
        else:
            message_type = ChatResponseMessage
            response_tool_calls = None
    
        response_message = message_type(
            role='assistant', 
            content=response.message.content,
            message=response.message,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            total_tokens=response.prompt_eval_count + response.eval_count,
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

    def get_num_ctx(self, model):
        token_limits = {
            128000: {
                "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
                "qwen2:7b", "qwen2:72b", "qwen3:8b",
                "hermes3:3b", "hermes3:8b", "hermes3:70b",
                "llama3.1", "llama3.1:8b", "llama3.1:70b",
                "llama3.3", "llama3.3:70b", "llama3.2:1b", "llama3.2:3b",
                "llama3-groq-tool-use", "llama3-groq-tool-use:8b", "llama3-groq-tool-use:70b"
            },
            32000: {
                "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwq", "qwq:32b",
                "mistral-small:22b", "mistral-large:123b", "mistral-small:24b"
            }
        }

        # Check which token limit applies, return default (128000) if not found
        for max_tokens, models in token_limits.items():
            if model in models:
                return max_tokens

        return 128000  # Default case