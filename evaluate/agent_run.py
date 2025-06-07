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
        self.agent_run_result = None

    def update_agent_run_result(self, task_result):
        # task_result is AgentTask instance!
        self.agent_run_result = task_result.to_dict()
        if hasattr(self.platform.model_client, "to_dict"):
            self.agent_run_result["model_client"] = self.platform.model_client.to_dict()
        # if self.platform.database.images_gdf:
        #     ... save

    def save_agent_run_result(self, task_result):
        """
        Saves the collected solution (a dict) to the results_output_file.
        The overall solution is structured as:
        {
            "tasks": [ <serialized AgentInferenceResult dict>, ... ],
            "model_client": <platform.model_client.to_dict()>
        }
        """
        self.update_agent_run_result(task_result)
        try:
            with open(self.results_output_file, "w") as f:
                json.dump(self.agent_run_result, f, indent=2)
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
