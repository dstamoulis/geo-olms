# tasks/geo_task.py

class AgentTask:
    def __init__(self, queries=None, rounds=None):
        """
        Initialize an AgentTask.
        
        Args:
            queries (list of str): Optionally, a list of initial queries.
            rounds (list of dict): Optionally, a list of rounds already completed.
                Each round is a dict with keys like 'query' and 'messages'.
        """
        self.queries = queries if queries is not None else []
        # The rounds will be a simple list of rounds.
        self.rounds = rounds if rounds is not None else []

    def add_round(self, query, messages):
        """
        Adds a new round (query + response) to the task.
        
        Args:
            query (str): The new query.
            messages (list): The conversation messages after processing the query.
        """
        self.queries.append(query)
        self.rounds.append({"query": query, "messages": messages})

    def get_full_conversation(self):
        """
        Returns the full conversation history (all messages from all rounds).
        """
        full_conversation = []
        for round_data in self.rounds:
            full_conversation.extend(round_data["messages"])
        return full_conversation

    def to_dict(self):
        """
        Serializes the AgentTask to a dictionary.
        """
        return {
            "queries": self.queries,
            "rounds": self.rounds
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes a dictionary into an AgentTask instance.
        
        Args:
            data (dict): A dictionary with keys "queries" and "rounds".
        
        Returns:
            AgentTask: A new instance.
        """
        queries = data.get("queries", [])
        rounds = data.get("rounds", [])
        return cls(queries, rounds)
