import json
from collections import defaultdict
import numpy as np
import fiona
import geopandas as gpd
import os


class AgentMetrics:

    def __init__(self, gts_path, results_path):

        self.gts_path = gts_path
        self.results_path = results_path
        self.reset_metrics()
        self.verbose = False

    def reset_metrics(self):

        self.allowed_functions = set()

        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.total_runtime = 0
        self.matched_queries = 0
        self.num_runs_evaluated = 0

        self.avg_total_tokens = 0
        self.avg_prompt_tokens = 0
        self.avg_completion_tokens = 0
        self.avg_cached_tokens = 0
        self.avg_runtime = 0

        self.correctness_list = []
        self.error_types = defaultdict(int)
        self.avg_correctness = 0
        self.error_details_dict = {}

        self.success_list = []
        self.avg_success = 0
        self.unsuccess_types = defaultdict(int)

        self.missing_runs = []


    def extend_allowed_functions(self, gt_tool_calls):

        for call in gt_tool_calls:
            self.allowed_functions.add(call.get('name'))
    

    @staticmethod
    def check_argument_error(gt_call, candidate_call):
        """
        Compare arguments between a ground truth tool call and a candidate call.
        Returns True if an error is detected.
        """
        try:
            gt_args = json.loads(gt_call.get('arguments', '{}'))
        except Exception:
            gt_args = {}
        try:
            candidate_args = json.loads(candidate_call.get('arguments', '{}'))
        except Exception:
            candidate_args = {}

        for arg, value in gt_args.items():
            if arg not in candidate_args:
                return True  # Missing argument.
            # For numeric types, allow a small tolerance.
            if isinstance(value, (int, float)) and candidate_args.get(arg) is not None:
                if abs(float(candidate_args[arg]) - float(value)) > 2:
                    return True
            # For strings, ensure candidate has a nonempty value.
            elif isinstance(value, str) and not candidate_args.get(arg):
                return True
        return False


    def correctness(self, gts_dict, results_dict):
        """
        Computes correctness per task and then averages.
        Also accumulates error types including:
            - Infeasible Action (candidate's tool not in allowed functions)
            - Function Error (tool names do not match)
            - Argument Error (arguments don't match)
            - Missed Function (ground truth tool call was missing)
            - Redundant Step Error (extra candidate calls)

        """ 

        # Get tool calls from ground truth
        gt_tool_calls = []
        for msg in gts_dict.get('messages', []):
            if msg.get('tool_calls'):
                gt_tool_calls.extend(msg['tool_calls'])

        self.extend_allowed_functions(gt_tool_calls)
            
        # Get tool calls from run result
        result_tool_calls = []
        for msg in results_dict.get('messages', []):
            if msg.get('tool_calls'):
                result_tool_calls.extend(msg['tool_calls'])

        num_gt_calls = len(gt_tool_calls)
        errors_in_result = 0
        used_candidate_indices = set()
        
        # For each ground truth call, try to find a matching candidate call by name that hasn't been used.
        for gt_tool_call in gt_tool_calls:
            match_index = None
            for idx, result_tool_call in enumerate(result_tool_calls):
                if idx in used_candidate_indices:
                    continue
                # handle handoff logic (steps)
                if "handoff_transfer_to_" in result_tool_call.get('name'):
                    # match_index = idx
                    used_candidate_indices.add(idx)
                    continue
                # Candidate's function must be allowed.
                if result_tool_call.get('name') not in self.allowed_functions:
                    # If candidate is not allowed, count it as infeasible and mark it as used.
                    self.error_types["Infeasible Action"] += 1
                    errors_in_result += 1
                    used_candidate_indices.add(idx)
                    continue
                if result_tool_call.get('name') == gt_tool_call.get('name'):
                    match_index = idx
                    break
            if match_index is None:
                self.error_types["Missed Function"] += 1
                errors_in_result += 1
            else:
                used_candidate_indices.add(match_index)
                result_tool_call = result_tool_calls[match_index]
                # Compare arguments using your existing check.
                if self.check_argument_error(gt_tool_call, result_tool_call):
                    self.error_types["Argument Error"] += 1
                    errors_in_result 

        # Any candidate calls not used are counted as redundant.
        extra_calls = len(result_tool_calls) - len(used_candidate_indices)
        if extra_calls > 0:
            self.error_types["Redundant Step Error"] += extra_calls
            errors_in_result += extra_calls
        
        # Compute correctness for this task (clip at 0).
        result_correctness = max(1 - (errors_in_result / num_gt_calls), 0) if num_gt_calls > 0 else 0
        self.correctness_list.append(result_correctness)
        self.avg_correctness = np.mean(self.correctness_list) if self.correctness_list else -1


    def compute_avg_llm_metrics(self):
            
        # Compute averages over number of runs that had (valid) results.
        if self.num_runs_evaluated == 0: return
        self.avg_total_tokens = self.total_tokens / self.num_runs_evaluated
        self.avg_prompt_tokens = self.total_prompt_tokens / self.num_runs_evaluated
        self.avg_completion_tokens = self.total_completion_tokens / self.num_runs_evaluated
        self.avg_cached_tokens = self.total_cached_tokens / self.num_runs_evaluated
        self.avg_runtime = self.total_runtime / self.num_runs_evaluated
        

    def print_avg_llm_metrics(self):
        """
        Prints the average LLM metrics collected over all runs
        in a nicely formatted table.
        """
        # Collect the values into a list of (label, value) pairs
        rows = [
            ("Runs evaluated",        self.num_runs_evaluated),
            ("Avg total tokens",      self.avg_total_tokens),
            ("Avg prompt tokens",     self.avg_prompt_tokens),
            ("Avg completion tokens", self.avg_completion_tokens),
            ("Avg cached tokens",     self.avg_cached_tokens),
            ("Avg runtime (s)",       self.avg_runtime),
        ]

        # Determine column widths
        label_width = max(len(label) for label, _ in rows) + 2
        value_width = 12

        # Header
        print("\nLLM Metrics Summary")
        print("-" * (label_width + value_width))
        # Rows
        for label, value in rows:
            # If it's an integer, print without decimals; otherwise two decimals
            if isinstance(value, int):
                val_str = f"{value}"
            else:
                val_str = f"{value:.2f}"
            print(f"{label:<{label_width}}{val_str:>{value_width}}")
        print("-" * (label_width + value_width), "\n")


    def print_error_counts(self):
        """
        Prints the counts for each error type in a formatted table.
        """
        error_keys = [
            "Infeasible Action",
            "Missed Function",
            "Argument Error",
            "Redundant Step Error"
        ]
        # Build rows of (label, count)
        rows = [(key, self.error_types.get(key, 0)) for key in error_keys]

        # Compute column widths
        label_width = max(len(label) for label, _ in rows) + 2
        value_width = max(len(str(count)) for _, count in rows) + 2

        # Print header
        print("\nError Types (Count)")
        print("-" * (label_width + value_width))

        # Print each error count
        for label, count in rows:
            print(f"{label:<{label_width}}{count:>{value_width}}")

        print("-" * (label_width + value_width), "\n")

    def print_unsuccess_counts(self):
        """
        Prints the counts for each unssuccess issue in a formatted table.
        """
        error_types_dict = {
            "images layers": "Missing Images Dataset (not loaded)",
            "detections layers": "Missing Detector (not ran)",
            " layers": "Missing layers",
            "images items": "Missing Images in Dataset (loaded but missing)",
            "detections items": "Missing Detections (Detection run but missing dets)",
            " items": "Missing items",
        }
        error_types_dict = {
            "images layers": "Missing Images (Dataset not loaded)",
            "detections layers": "Missing Detections (No Detector run)",
            "images items": "Missing Images in Dataset (loaded but missing)",
            "detections items": "Missing Detections (Detection run but missing dets)",
            "images scatter": "Missing images scatter plot",
            "detections scatter": "Missing detections scatter plot",
            "landcover scatter": "Missing landcover scatter plot",
            "map zoom": "Map not properly zoomed",
        }

        # Build rows of (label, count)
        rows = [(error_types_dict[key], self.unsuccess_types.get(key, 0)) for key in error_types_dict.keys()]

        # Compute column widths
        label_width = max(len(label) for label, _ in rows) + 2
        value_width = max(len(str(count)) for _, count in rows) + 2

        # Print header
        print("Unsuccess Reason Types (Count)")
        print("-" * (label_width + value_width))

        # Print each error count
        for label, count in rows:
            print(f"{label:<{label_width}}{count:>{value_width}}")

        print("-" * (label_width + value_width), "\n")


    def print_avg_agent_scores(self):
        """
        Prints the average correctness and average success side by side
        in a single, neatly formatted table.
        """
        # Prepare the two metrics
        metrics = [
            ("Overall Correctness", self.avg_correctness),
            ("Overall Success", self.avg_success)
        ]

        # Format values and build rows
        rows = []
        for label, val in metrics:
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                val_str = f"{val:.1%}"
            else:
                val_str = str(val)
            rows.append((label, val_str))

        # Compute column widths
        label_width = max(len(label) for label, _ in rows) + 2
        value_width = max(len(value) for _, value in rows) + 2

        # Print header
        print("\nAgent Scores Summary")
        print("-" * (label_width + value_width))

        # Print each metric
        for label, val_str in rows:
            print(f"{label:<{label_width}}{val_str:>{value_width}}")

        # Footer
        print("-" * (label_width + value_width), "\n")



    def llm_metrics(self, results_dict):
        """
        Computes average system metrics (runtime, tokens) per query (for queries that matched
        ground truth).
        
        This function iterates over the agent result tasks corresponding to the ground truth,
        and sums metrics from each message (e.g., prompt_tokens, completion_tokens, cached_tokens, time_elapsed).
        """

        messages = results_dict.get("messages", [])
        if not messages: return
        self.num_runs_evaluated += 1
        # Sum metrics from each message in the round.
        for msg in messages:
            self.total_prompt_tokens += msg.get("prompt_tokens") or 0
            self.total_completion_tokens += msg.get("completion_tokens") or 0
            self.total_cached_tokens += msg.get("cached_tokens") or 0
            self.total_runtime += msg.get("time_elapsed") or 0
            self.total_tokens += (msg.get("prompt_tokens") or 0) + (msg.get("completion_tokens") or 0)

        self.compute_avg_llm_metrics()


    def uoi_success(self, gt_uois_dict: dict, res_uois_dict: dict, uoi_type: str = "", error_tol: float = 0.0) -> int:
        """
        Compares two UOI lists (ground-truth vs. results) for "success" under the following rules:
        
        1. Every layer in the GT must be present in the results (i.e. results_layers ⊇ gt_layers).
        2. For each GT layer, compare the set of unique `uoi` values:
        - Let GT = set of uoi in the GT layer.
        - Let RES = set of uoi in the same-named results layer.
        - It's a pass if |GT \ RES| / |GT| ≤ error_tol.
            (i.e. you may miss up to error_tol fraction of GT images)
        
        Args:
            gt_uois_dict: Dict with ground-truth UOIs json.
            res_uois_dict: Dict with results UOIs json.
            uoi_type: Str either (images or detections)
            error_tol: Fractional tolerance for missing items (0 ⇒ must have 100% of GT).
        
        Returns:
            1 if both checks pass, else 0.
        """
        # 1. List layers
        gt_layers  = set(list(gt_uois_dict.keys()))
        res_layers = set(list(res_uois_dict.keys()))

        success = True
        
        # Check 1: all GT layers must be in results
        if not gt_layers.issubset(res_layers):
            missing = gt_layers - res_layers
            if self.verbose: print(f"Missing layers in results: {missing}")
            self.unsuccess_types[f"{uoi_type} layer"] +=1
            success = False
        
        # Check 2: per-layer uoi superset with tolerance
        for layer in gt_layers:
            
            gt_uois  = set(gt_uois_dict.get(layer, []))
            res_uois = set(res_uois_dict.get(layer, []))
            
            # fraction missing = |GT - RES| / |GT|
            missing_count = len(gt_uois - res_uois)
            frac_missing  = missing_count / len(gt_uois) if gt_uois else 0.0
            
            if frac_missing > error_tol:
                if self.verbose : 
                    print(f"Layer '{layer}': missing {missing_count}/{len(gt_uois)} "
                        f"({frac_missing:.1%} > tol={error_tol:.1%})")
                self.unsuccess_types[f"{uoi_type} items"] +=1
                success = False
        
        # All checks passed
        return success


    def map_state_success(self, gt_map_dict, results_map_dict, center_tol=0.5):
        """
        Compare a ground-truth map state dict and a result map state dict for “success”:
        1. All three lists under gt["map_state"] must match exactly res["map_state"] (as sets).
        2. The absolute difference in center lat/lon must be ≤ center_tol.    
        Returns 1 if both conditions pass, else 0.
        """
        map_error_types = {
            "image_datasets_scatter_plots" : "images scatter",
            "detections_scatter_plots" : "detections scatter",
            "landcover_scatter_plots" : "landcover scatter"
        }
        success = True
        gt_state = gt_map_dict.get("map_state", {})
        res_state = results_map_dict.get("map_state", {})
        for key, plot_error_type in map_error_types.items():
            gt_list = gt_state.get(key, [])
            res_list = res_state.get(key, [])
            if set(gt_list) != set(res_list):
                self.unsuccess_types[plot_error_type] +=1
                success = False

        try:
            gt_center = gt_map_dict.get("map_layout", {})\
                        .get("mapbox", {})\
                        .get("center", {})
            res_center = results_map_dict.get("map_layout", {})\
                            .get("mapbox", {})\
                            .get("center", {})
            gt_lat, gt_lon = float(gt_center["lat"]), float(gt_center["lon"])
            res_lat, res_lon = float(res_center["lat"]), float(res_center["lon"])

            if abs(gt_lat - res_lat) > center_tol or abs(gt_lon - res_lon) > center_tol:
                if self.verbose: print(f"Center mismatch: GT=({gt_lat},{gt_lon}), "
                        f"RES=({res_lat},{res_lon}), tol={center_tol}")
                self.unsuccess_types["map zoom"] +=1
                success = False

        except (KeyError, TypeError, ValueError):
            print("Invalid or missing center coordinates in map_layout.mapbox.center")
            
        return success

    def success(self, 
            gts_dict, results_dict,
            gt_images_uois_dict, results_images_uois_dict, 
            gt_detections_uois_dict, results_detections_uois_dict, 
            gt_map_dict, results_map_dict,
            imgs_error_tol = 0.2, dets_error_tol=0.2, map_degrees_error_tol=0.2
        ):

        images_success = self.uoi_success(
            gt_images_uois_dict, results_images_uois_dict,
            uoi_type="images", error_tol = imgs_error_tol
            )
        detections_success = self.uoi_success(
            gt_detections_uois_dict, results_detections_uois_dict,
            uoi_type="detections", error_tol = dets_error_tol
            )
        map_success = self.map_state_success(gt_map_dict, results_map_dict, map_degrees_error_tol)

        successful_run = images_success and detections_success and map_success
        self.success_list.append(successful_run)
        self.avg_success = np.mean(self.success_list) if self.success_list else -1

    def missing_run(self, run_id):
        self.missing_runs.append(run_id)
        self.success_list.append(False)
        self.avg_success = np.mean(self.success_list) if self.success_list else -1

    def missing_runs_stats(self):
        print(f"Total: {len(self.missing_runs)}. Missing runs: {self.missing_runs}")

    def evaluate_run(self, run_id):

        path = os.path.join(self.results_path, str(run_id), "result.json")
        if not os.path.isfile(path):
            self.missing_run(run_id)
            return

        with open(os.path.join(self.gts_path, str(run_id), "result.json")) as f:
            gts_dict = json.load(f)
        with open(os.path.join(self.results_path, str(run_id), "result.json")) as f:
            results_dict = json.load(f)

        with open(os.path.join(self.gts_path, str(run_id), "images_gdf.json")) as f:
            gt_images_uois_dict = json.load(f)
        with open(os.path.join(self.results_path, str(run_id), "images_gdf.json")) as f:
            results_images_uois_dict = json.load(f)

        with open(os.path.join(self.gts_path, str(run_id), "detections_gdf.json")) as f:
            gt_detections_uois_dict = json.load(f)
        with open(os.path.join(self.results_path, str(run_id), "detections_gdf.json")) as f:
            results_detections_uois_dict = json.load(f)

        with open(os.path.join(self.gts_path, str(run_id), "map_state.json")) as f:
            gt_map_dict = json.load(f)
        with open(os.path.join(self.results_path, str(run_id), "map_state.json")) as f:
            results_map_dict = json.load(f)

        self.correctness(gts_dict, results_dict)
        self.success(
            gts_dict, results_dict,
            gt_images_uois_dict, results_images_uois_dict, 
            gt_detections_uois_dict, results_detections_uois_dict, 
            gt_map_dict, results_map_dict
        )
        self.llm_metrics(results_dict)



# Example usage:
if __name__ == "__main__":

    gts_prefix = "results/openai/gpt_4o_mini/single_agent/openai_gpt_4o_mini_0_1_single_agent"
    results_prefix = "results/openai/gpt_4o_mini/geoflow/openai_gpt_4o_mini_0_1_geoflow"
    agent_metrics = AgentMetrics(gts_path, results_path)

    runs_from, runs_to = 0, 3
    runs_id = [i for i in range(runs_from, runs_to)]

    for run_id in runs_id:
        agent_metrics.evaluate_run(run_id)

    agent_metrics.print_avg_llm_metrics()
