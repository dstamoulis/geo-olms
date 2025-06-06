# benchmark/agent_benchmark.py

import json, os
from evaluate.agent_task import AgentTask

class AgentRun:
    def __init__(self, platform, results_output_file):
        """
        Initialize the benchmark with an platform, dataset file, and output file.
        
        Args:
            platform: An instance of your platform.
            results_output_file (str): Path where the solution will be saved.
        """
        self.platform = platform

        self.results_output_file = results_output_file
        os.makedirs('results', exist_ok=True) # TODO: do this properly

        # List of AgentTask instances.
        self.benchmark_tasks = []
        self.agent_tasks_results = []

    def add_task_result(self, task_result):
        # task_result is AgentTask instance!
        self.agent_tasks_results.append(task_result)

    def load_dataset(self, benchmark_input_file):
        """
        Loads multi-round tasks from a JSON file. 
        Expects a structure like:
        {
            "tasks": [
                {"queries": ["query1", "query2", ...]},
                ...
            ]
        }
        """
        try:
            with open(benchmark_input_file, "r") as f:
                data = json.load(f)
            tasks_data = data.get("tasks", [])
            for task_data in tasks_data:
                # Create an AgentInferenceResult for each entry.
                benchmark_task = AgentTask(queries=task_data.get("queries", []))
                self.benchmark_tasks.append(benchmark_task)
            print(f"Loaded {len(self.benchmark_tasks)} multi-round tasks from dataset.")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {benchmark_input_file}: {e}")

    def run_all(self):
        """
        Iterate over all benchmark tasks and run each query sequentially.
        For each query in a task, the agent's conversation is updated and then recorded.
        """
        for idx, task in enumerate(self.benchmark_tasks):
            print(f"Running benchmark task {idx+1}/{len(self.benchmark_tasks)}")
            # Reset or initialize agent state as needed between tasks.
            # If you want to keep conversation across tasks, omit this step.
            self.platform.reset()

            add_task_result = AgentTask()
            
            for query in task.queries:
                print(f"Running query: {query}")
                # Capture the number of existing messages before running this query.
                num_current_messages = len(self.platform.messages)
                # Run the query (agent should update its messages internally).
                response = self.platform.run_query(query)
                # # Record this round in the AgentInferenceResult.
                add_task_result.add_round(query, self.platform.messages.to_list_dict(start=num_current_messages))
            
            # # Append the task (serialized) to the overall solution.
            self.add_task_result(add_task_result)
        
        # # Optionally store model client info.
        # self.solution["model_client"] = self.platform.model_client.to_dict()
        # print("Multi-round benchmark run complete.")

    def run_from_benchmark_file(self, benchmark_input_file):
        # TODO: add check the benchmark_input_file exists!
        self.load_dataset(benchmark_input_file)
        self.run_all()

    def save_inference_results(self):
        """
        Saves the collected solution (a dict) to the results_output_file.
        The overall solution is structured as:
        {
            "tasks": [ <serialized AgentInferenceResult dict>, ... ],
            "model_client": <platform.model_client.to_dict()>
        }
        """
        inference_results = {
            "tasks": [agent_tasks_result.to_dict() for agent_tasks_result in self.agent_tasks_results]
        }
        if hasattr(self.platform.model_client, "to_dict"):
            inference_results["model_client"] = self.platform.model_client.to_dict()
        try:
            with open(self.results_output_file, "w") as f:
                json.dump(inference_results, f, indent=2)
            print(f"Inference results saved to {self.results_output_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to save inference results to {self.results_output_file}: {e}")

    # @classmethod
    # def from_cfg(cls, agent, cfg: dict):
    #     """
    #     Create a Evaluate instance from a configuration dictionary.
        
    #     Expected keys in cfg:
    #       - dataset_path: path to the dataset JSON.
    #       - results_output_file: path for saving the solution.
    #     """
    #     dataset_path = cfg.get("dataset_path")
    #     results_output_file = cfg.get("results_output_file")
    #     if not dataset_path or not results_output_file:
    #         raise ValueError("Benchmark config must include 'dataset_path' and 'results_output_file'.")
    #     return cls(agent, dataset_path, results_output_file)
