# llm_clients/base_client.py

class BaseClient:
    client_class: str = "BaseClient"

    def __init__(self, model, temperature, **kwargs):
        """
        Initialize the base LLM client.

        Args:
            model (str): The name of the model to use.
            **kwargs: Additional keyword arguments (e.g., API keys).
        """
        self.model = model
        self.temperature = temperature
        # Store any additional initialization parameters (e.g., api_key)
        self.params = kwargs
        self.client_class = self.__class__.client_class

    def get_response(self, messages, tools=None):
        """
        Given a conversation history (messages) and an optional list of tools,
        return a response from the language model.
        
        Args:
            messages (list): A list of message dictionaries representing the conversation history.
            tools (list, optional): A list of tool specifications available for function calling.
        
        Returns:
            dict: A dictionary containing the model's response, input tokens, output tokens, etc.
        """
        raise NotImplementedError("get_response() must be implemented by subclasses.")

    def client_info(self):
        return {
            'client_class': self.client_class,
            'model': self.model
        }
    
    @classmethod
    def from_cfg(cls, cfg: dict) -> "BaseClient":
        """
        Create an instance of a client based on the provided configuration.
        
        The configuration is expected to include at least:
            - client: a string indicating the client type ("openai", "ollama", or "vllm").
            - model: the model name.
            - temperature: the temperature parameter (with a default provided if missing).
            - Any additional keys will be forwarded as keyword arguments.
        
        Args:
            cfg (dict): Configuration dictionary.
        
        Returns:
            BaseClient: An instance of the appropriate client subclass.
        """
        client_type = cfg.get("client", "openai").lower()
        model = cfg.get("model", "gpt-4o-mini")
        temperature = cfg.get("temperature", 0.1)
        # Gather any additional parameters.
        extra_params = {k: v for k, v in cfg.items() if k not in ["client", "model", "temperature"]}

        # NOTE:  This could also be a standalone factory function, for example
        # `create_llm_endpoint(cfg: dict) -> BaseClient` but don't really like factories
        if client_type == "openai":
            from .openai_client import OpenAIClient
            return OpenAIClient(model, temperature, **extra_params)
        elif client_type == "ollama":
            from .ollama_client import OllamaClient
            return OllamaClient(model, temperature, **extra_params)
        elif client_type == "vllm":
            from .vllm_client import VLLMClient
            return VLLMClient(model, temperature, **extra_params)
        else:
            raise ValueError(f"Unsupported client: {client_type}")