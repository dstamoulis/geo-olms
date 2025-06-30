# benchmark/agent_benchmark.py

import json, os
from evaluate.agent_task import AgentTask

class AgentRun:
    
    def __init__(self, platform, results_output_filenames):
        """
        Initialize the benchmark with an platform, dataset file, and output file.
        
        Args:
            platform: An instance of your platform.
            results_output_filenames (dict): Path where the solution will be saved.
        """
        self.platform = platform
        self.results_output_filenames = results_output_filenames
        self.agent_run_result = None

    def update_agent_run_result(self, task_result):
        # task_result is AgentTask instance!
        self.agent_run_result = task_result.to_dict()
        if hasattr(self.platform.model_client, "to_dict"):
            self.agent_run_result["model_client"] = self.platform.model_client.to_dict()
        # if self.platform.database.images_gdf:
        #     ... save

    def save_images_gdf_uois(self, gdfs_dict: dict, output_json_path: str):
        """
        Given a dict mapping layer names to GeoDataFrames, extract the 'uoi' column
        from each non-empty GeoDataFrame and save as a JSON mapping layer → [uoi, ...].

        Args:
            gdfs_dict (dict): { layer_name (str): GeoDataFrame, … }
            output_json_path (str): path to write the resulting JSON.
        """
        # Build the mapping
        uoi_map = {}
        for layer, gdf in gdfs_dict.items():
            if gdf is None or gdf.empty:
                # skip empty or missing
                continue
            # Extract the uoi column as a list
            uoi_map[layer] = gdf["uoi"].astype(str).tolist()
        # Write out pretty-printed JSON
        with open(output_json_path, "w") as f:
            json.dump(uoi_map, f, indent=2)

        print(f"Saved images UOIs to {output_json_path}")


    def save_detections_gdf_uois(self, gdfs_dict: dict, output_json_path: str):
        """
        Given a dict mapping layer names to GeoDataFrames, extract the 'uoi' column
        from each non-empty GeoDataFrame and save as a JSON mapping layer → [uoi, ...].

        Args:
            gdfs_dict (dict): { layer_name (str): GeoDataFrame, … }
            output_json_path (str): path to write the resulting JSON.
        """
        # Build the mapping
        uoi_map = {}
        for dataset, models in gdfs_dict.items():
            for model_name, gdf in models.items():
                if gdf is None or gdf.empty:
                    # Skip empty or missing GeoDataFrames
                    continue
                layer = f"{dataset}_{model_name}".replace(" ", "_")
                # Extract the uoi column as a list
                uoi_map[layer] = gdf["uoi"].astype(str).tolist()

        # Write out pretty-printed JSON
        with open(output_json_path, "w") as f:
            json.dump(uoi_map, f, indent=2)

        print(f"Saved detections UOIs to {output_json_path}")


    def save_map_state(self, map_state: dict, output_json_path):
        """
        Combine map_state dict with just the mapbox & margin from a Plotly Figure
        into a JSON‐serializable dict.
        
        Args:
            map_state: your existing map_state, e.g.
                {'image_datasets_scatter_plots': [...], ...}
            fig: a plotly.graph_objects.Figure with a layout.mapbox and layout.margin
            
        Returns:
            {
            "map_state": { … },
            "map_layout": {
                "mapbox": { "center": {...}, "zoom": ..., "style": ..., ... },
                "margin": { "l": ..., "r": ..., "t": ..., "b": ... }
            }
            }
        """
        fig = map_state['map_layout'] 
        # Convert the full figure to a plain dict and grab layout
        full_layout = fig.to_dict().get("layout", {})
        
        # Extract only the parts you care about
        mapbox = full_layout.get("mapbox", {})
        margin = full_layout.get("margin", {})
        
        map_state_dict = {
            "map_state": map_state['map_state'] ,
            "map_layout": {
                "mapbox": mapbox,
                "margin": margin
            }
        }
        with open(output_json_path, "w") as f:
            json.dump(map_state_dict, f, indent=2)
        print(f"Saved map state to {output_json_path}")


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
            with open(self.results_output_filenames["result"], "w") as f:
                json.dump(self.agent_run_result, f, indent=2)
            self.save_images_gdf_uois(self.platform.database.images_gdf, self.results_output_filenames["images_gdf"])
            self.save_detections_gdf_uois(self.platform.vision.detections_gdf, self.results_output_filenames["detections_gdf"])
            self.save_map_state(self.platform.map_ops.get_map_state(), self.results_output_filenames["map_state"])
            print(f"Inference results saved to {self.results_output_filenames['results_path']}")
        except Exception as e:
            raise RuntimeError(f"Failed to save inference results to {self.results_output_filenames['results_path']}: {e}")


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
