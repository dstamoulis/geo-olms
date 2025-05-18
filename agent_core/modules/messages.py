# messages.py

class BaseMessage:
    msg_class: str = "BaseMessage"

    def __init__(
            self, 
            role, 
            content="", 
            message=None, 
            prompt_tokens=None, 
            cached_tokens=None, 
            completion_tokens=None, 
            total_tokens=None, 
            time_elapsed=None, 
            tool_calls=None, 
            source=None
        ):
        """
        BaseMessage holds common attributes for all message types.
        
        Args:
            role (str): The role for the message (e.g., 'user', 'assistant', 'tool').
            content (str, optional): The message text.
            prompt_tokens (int, optional): The number of input tokens.
            cached_tokens (int, optional): The number of cached tokens.
            completion_tokens (int, optional): The number of output tokens.
            tool_calls (list, optional): A list of ToolCall objects, if any.
            source (str, optional): A source identifier (if applicable).
        """
        self.role = role
        self.content = content
        self.message = message
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.time_elapsed = time_elapsed
        self.tool_calls = tool_calls  # Expected to be a list of ToolCall objects
        self.source = source
        self.msg_class = self.__class__.msg_class

    def __str__(self):
        parts = [f"Role: {self.role}"]
        if self.content:
            parts.append(f"Content: {self.content}")
        if self.prompt_tokens is not None:
            parts.append(f"Input Tokens: {self.prompt_tokens}")
        if self.cached_tokens is not None:
            parts.append(f"Cached Tokens: {self.cached_tokens}")
        if self.completion_tokens is not None:
            parts.append(f"Output Tokens: {self.completion_tokens}")
        if self.tool_calls:
            parts.append(f"Tool Calls: {[str(tc) for tc in self.tool_calls]}")
        if self.source:
            parts.append(f"Source: {self.source}")
        if self.time_elapsed is not None:
            parts.append(f"Time Elapsed: {self.time_elapsed}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        result = {
            "msg_class": self.msg_class,
            "role": self.role,
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "time_elapsed": self.time_elapsed,
            "source": self.source,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
        }
        if self.message is not None:
            result["message"] = str(self.message)  # Customize if your 'message' has its own serialization.
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class TextMessage(BaseMessage):
    msg_class: str = "TextMessage"

    def __init__(self, role, content="", **kwargs):
        """
        TextMessage represents a plain text message, e.g. user input to LLM
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"TextMessage: {super().__str__()}"


class ChatResponseMessage(BaseMessage):
    msg_class: str = "ChatResponseMessage"

    def __init__(self, role, content="", **kwargs):
        """
        ChatResponseMessage represents a plain text chat response.
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"ChatResponseMessage: {super().__str__()}"


class ToolCallRequestMessage(BaseMessage):
    msg_class: str = "ToolCallRequestMessage"
    
    def __init__(self, role, content="", **kwargs):
        """
        ToolCallRequestMessage is used when the assistant is requesting a tool call.
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"ToolCallRequestMessage: {super().__str__()}"


class UserProgramToolCallRequest(BaseMessage):
    msg_class: str = "UserProgramToolCallRequest"
    
    def __init__(self, role, content="", **kwargs):
        """
        UserProgramToolCallRequest is used when the user script is explicitly requesting a tool call.
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"UserProgramToolCallRequest: {super().__str__()}"


class ToolResponseMessage(BaseMessage):
    msg_class: str = "ToolResponseMessage"
    
    def __init__(self, role, content="", **kwargs):
        """
        ToolResponseMessage is used when a tool returns a response.
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"ToolResponseMessage: {super().__str__()}"


class HandoffMessage(BaseMessage):
    msg_class: str = "HandoffMessage"
    
    def __init__(self, role, content="", **kwargs):
        """
        HandoffMessage is used when a tool returns an Agent to hand execution off to.
        """
        super().__init__(role, content, **kwargs)
    
    def __str__(self):
        return f"HandoffMessage: {super().__str__()}"


class ToolCall:
    def __init__(self, id, name, arguments, tool_type):
        """
        Represents a single tool call.
        
        Args:
            id (str): An identifier for the tool call.
            name (str): The name of the tool.
            arguments (str): A string (or JSON) representation of the arguments.
            tool_type (str): The type of tool, e.g., "function".
        """
        self.id = id
        self.name = name
        self.arguments = arguments
        self.tool_type = tool_type

    def __str__(self):
        return f"ToolCall(id={self.id}, name={self.name}, arguments={self.arguments}, type={self.tool_type})"

    def to_dict(self) -> dict:
        return {
            "type": "ToolCall",
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "tool_type": self.tool_type
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class Messages:
    def __init__(self):
        """
        Holds a list of messages for a conversation.
        """
        self.messages = []

    def add_message(self, message: BaseMessage):
        """
        Add a message to the conversation.
        
        Args:
            message (BaseMessage): A message instance.
        """
        self.messages.append(message)

    def reset_messages(self):
        """
        Clears all stored messages.
        """
        self.messages = []

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        """Allow iteration over the underlying messages list."""
        return iter(self.messages)
    
    def __getitem__(self, index):
        """Allow slicing/indexing on the underlying messages list."""
        return self.messages[index]

    def to_list(self):
        return self.messages

    def _get_client_messages(self):
        """
        Returns the conversation messages in a format suitable for LLM API calls.
        For example, it may loop through self.messages and convert each to a dict.
        (Placeholder implementation.)
        """
        api_messages = []
        for msg in self.messages:
            # Example conversion; extend this as needed for function calling, token info, etc.
            api_messages.append({
                "role": msg.role,
                "content": msg.content
                # Add token counts or tool_call details as needed.
            })
        return api_messages

    def get_openai_client_messages(self):
        """
        Returns the conversation messages in a format suitable for GPT API calls.
        For example, it may loop through messages (TextMessage, ChatResponseMessage) 
        and convert each to the {'role': ..., } format OpenAI API expects.
        """
        api_messages = []
        for msg in self.messages:
            if msg.message is not None:
                api_msg = msg.message
            else:
                api_msg = {
                    "role": msg.role,
                    "content": msg.content
                }
            if isinstance(api_msg, list):
                api_messages.extend(api_msg)
            else:
                api_messages.append(api_msg)
        return api_messages


    def get_client_messages(self, client_class):
        """
        Wrapper function. Returns the conversation messages based on the client type.
        """
        if client_class == "OpenAIClient":
            return self.get_openai_client_messages()
        elif client_class == "OllamaClient":
            return self.get_openai_client_messages()
        elif client_class == "VLLMClient":
            raise ValueError(f"Unsupported client: {client_class}")
        else:
            return self._get_client_messages()        

    def __str__(self):
        return "\n".join(str(m) for m in self.messages)

    def to_list_dict(self, start=0, end=None):
        """
        Converts a slice of stored messages to a list of dictionaries.
        Args:
            start (int): The starting index for slicing.
            end (int, optional): The ending index for slicing.
        Returns:
            list: A list where each element is a dictionary representation of a message.
        """
        return [message.to_dict() for message in self.messages[start:end]]
