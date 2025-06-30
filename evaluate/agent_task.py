# tasks/geo_task.py

class AgentTask:
    def __init__(self, query=None, messages=None):
        """
        Initialize an AgentTask.
        
        Args:
            query (str): Optionally, query.
            messages (list of dict): Optionally, a list of messages so far.
        """
        self.query = query
        # The rounds will be a simple list of rounds.
        self.messages = messages if messages is not None else []

    def add_run(self, query, messages):
        """
        Adds a new run (query + response) to the task.
        
        Args:
            query (str): The new query.
            messages (list): The conversation messages after processing the query.
        """
        self.query = query
        self.messages = messages 

    def get_full_conversation(self):
        """
        Returns the full conversation history (all messages so far).
        """
        return self.messages

    def to_dict(self):
        """
        Serializes the AgentTask to a dictionary.
        """
        return {
            "query": self.query,
            "messages": self.messages
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes a dictionary into an AgentTask instance.
        
        Args:
            data (dict): A dictionary with keys "query" and "messages".
        
        Returns:
            AgentTask: A new instance.
        """
        query = data.get("query", None)
        messages = data.get("messages", [])
        return cls(query, messages)
