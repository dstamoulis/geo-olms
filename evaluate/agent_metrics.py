import json
from collections import defaultdict
import numpy as np


class AgentMetrics:

    def __init__(self, gts_prefix, results_prefix):

        self.gts_prefix = gts_prefix
        self.results_prefix = results_prefix
        self.reset_metrics()

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
        self.overall_correctness = 0
        self.error_details_dict = {}


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
                    errors_in_result += 1
        
        # Any candidate calls not used are counted as redundant.
        extra_calls = len(result_tool_calls) - len(used_candidate_indices)
        if extra_calls > 0:
            self.error_types["Redundant Step Error"] += extra_calls
            errors_in_result += extra_calls
        
        # Compute correctness for this task (clip at 0).
        result_correctness = max(1 - (errors_in_result / num_gt_calls), 0) if num_gt_calls > 0 else 0
        self.correctness_list.append(result_correctness)
        self.overall_correctness = np.mean(self.correctness_list) if self.correctness_list else -1


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


    def print_overall_correctness(self):
        """
        Prints the overall correctness as a single-line table.
        """
        # Format as percentage if it’s a float in [0,1], otherwise raw value
        val = self.overall_correctness
        if isinstance(val, float) and 0.0 <= val <= 1.0:
            val_str = f"{val:.1%}"
        else:
            val_str = str(val)

        label = "Overall correctness"
        # Compute widths
        label_width = len(label) + 2
        value_width = len(val_str) + 2

        # Print
        print("Overall Correctness")
        print("-" * (label_width + value_width))
        print(f"{label:<{label_width}}{val_str:>{value_width}}")
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


    def evaluate_run(self, gts_file, results_file):

        with open(gts_file) as f:
            gts_dict = json.load(f)
        with open(results_file) as f:
            results_dict = json.load(f)

        self.correctness(gts_dict, results_dict)
        self.llm_metrics(results_dict)



# Example usage:
if __name__ == "__main__":

    gts_prefix = "results/openai/gpt_4o_mini/single_agent/openai_gpt_4o_mini_0_1_single_agent"
    results_prefix = "results/openai/gpt_4o_mini/geoflow/openai_gpt_4o_mini_0_1_geoflow"
    agent_metrics = AgentMetrics(gts_prefix, results_prefix)

    runs_from, runs_to = 0, 3
    runs_id = [i for i in range(runs_from, runs_to)]

    for run_id in runs_id:
        gts_file = f"{gts_prefix}_{run_id}.json"
        results_file = f"{results_prefix}_{run_id}.json"
        agent_metrics.evaluate_run(gts_file, results_file)

    agent_metrics.print_avg_llm_metrics()
