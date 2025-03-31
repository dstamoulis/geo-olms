import json
from collections import defaultdict
import numpy as np

class AgentMetrics:
    def __init__(self, gts_file, agent_results_file):
        """
        Initialize with ground truth and agent result records.
        Both gts and agent_result are dictionaries with a "tasks" key.
        """

        with open(gts_file) as f:
            gts = json.load(f)
        with open(agent_results_file) as f:
            agent_result = json.load(f)

        self.gts = gts
        self.agent_result = agent_result
        
        # Build a set of allowed function names from ground truth.
        self.allowed_functions = set()
        for task in self.gts.get('tasks', []):
            for rnd in task.get('rounds', []):
                for msg in rnd.get('messages', []):
                    if msg.get('tool_calls'):
                        for call in msg['tool_calls']:
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
                if abs(float(candidate_args[arg]) - float(value)) > 0.1:
                    return True
            # For strings, ensure candidate has a nonempty value.
            elif isinstance(value, str) and not candidate_args.get(arg):
                return True
        return False

    def correctness(self):
        """
        Computes correctness per task and then averages.
        Also accumulates error types including:
            - Infeasible Action (candidate's tool not in allowed functions)
            - Function Error (tool names do not match)
            - Argument Error (arguments don't match)
            - Missed Function (ground truth tool call was missing)
            - Redundant Step Error (extra candidate calls)
        
        Returns:
            (overall_correctness, error_types_dict)
        """
        correctness_list = []
        error_types = defaultdict(int)
        
        # Iterate over tasks assuming both gts and agent_result have corresponding tasks.
        for gt_task, cand_task in zip(self.gts.get('tasks', []), self.agent_result.get('tasks', [])):

            # Aggregate all tool calls from all rounds for the ground truth
            gt_calls = []
            for rnd in gt_task.get('rounds', []):
                for msg in rnd.get('messages', []):
                    if msg.get('tool_calls'):
                        gt_calls.extend(msg['tool_calls'])
            
            # Aggregate all tool calls from all rounds for the candidate
            cand_calls = []
            for rnd in cand_task.get('rounds', []):
                for msg in rnd.get('messages', []):
                    if msg.get('tool_calls'):
                        cand_calls.extend(msg['tool_calls'])

            num_gt_calls = len(gt_calls)
            errors_in_task = 0
            used_candidate_indices = set()
            
            # For each ground truth call, try to find a matching candidate call by name that hasn't been used.
            for gt_call in gt_calls:
                match_index = None
                for idx, candidate_call in enumerate(cand_calls):
                    if idx in used_candidate_indices:
                        continue
                    # Candidate's function must be allowed.
                    if candidate_call.get('name') not in self.allowed_functions:
                        # If candidate is not allowed, count it as infeasible and mark it as used.
                        error_types["Infeasible Action"] += 1
                        errors_in_task += 1
                        used_candidate_indices.add(idx)
                        continue
                    if candidate_call.get('name') == gt_call.get('name'):
                        match_index = idx
                        break
                if match_index is None:
                    error_types["Missed Function"] += 1
                    errors_in_task += 1
                else:
                    used_candidate_indices.add(match_index)
                    candidate_call = cand_calls[match_index]
                    # Compare arguments using your existing check.
                    if self.check_argument_error(gt_call, candidate_call):
                        error_types["Argument Error"] += 1
                        errors_in_task += 1
            
            # Any candidate calls not used are counted as redundant.
            extra_calls = len(cand_calls) - len(used_candidate_indices)
            if extra_calls > 0:
                error_types["Redundant Step Error"] += extra_calls
                errors_in_task += extra_calls
            
            # Compute correctness for this task (clip at 0).
            task_correctness = max(1 - (errors_in_task / num_gt_calls), 0) if num_gt_calls > 0 else 0
            correctness_list.append(task_correctness)
        
        overall_correctness = np.mean(correctness_list) if correctness_list else -1
        return overall_correctness, dict(error_types)

    def system_metrics(self):
        """
        Computes average system metrics (runtime, tokens) per query (for queries that matched
        ground truth).
        
        This function iterates over the agent result tasks corresponding to the ground truth,
        and sums metrics from each message (e.g., prompt_tokens, completion_tokens, cached_tokens, time_elapsed).
        
        Returns:
            A dictionary containing average tokens (total, prompt, completion, cached) and average runtime.
        """
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_runtime = 0
        matched_queries = 0

        # Iterate over tasks (ground truth and candidate)
        for gt_task, cand_task in zip(self.gts.get("tasks", []), self.agent_result.get("tasks", [])):
            # Iterate over each round (each round corresponds to a query)
            for rnd in cand_task.get("rounds", []):
                messages = rnd.get("messages", [])
                if not messages:
                    continue
                matched_queries += 1
                # Sum metrics from each message in the round.
                for msg in messages:
                    total_prompt_tokens += msg.get("prompt_tokens") or 0
                    total_completion_tokens += msg.get("completion_tokens") or 0
                    total_cached_tokens += msg.get("cached_tokens") or 0
                    total_runtime += msg.get("time_elapsed") or 0
                    total_tokens += (msg.get("prompt_tokens") or 0) + (msg.get("completion_tokens") or 0)

        # Compute averages over the matched queries.
        if matched_queries > 0:
            avg_metrics = {
                "avg_total_tokens": total_tokens / matched_queries,
                "avg_prompt_tokens": total_prompt_tokens / matched_queries,
                "avg_completion_tokens": total_completion_tokens / matched_queries,
                "avg_cached_tokens": total_cached_tokens / matched_queries,
                "avg_runtime": total_runtime / matched_queries,
            }
        else:
            avg_metrics = {}

        return avg_metrics

    def eval_all(self):
        """
        Runs all evaluations and returns a summary dictionary.
        """
        correctness_rate, errors = self.correctness()
        metrics = self.system_metrics()
        return {
            "correctness": correctness_rate,
            "errors": errors,
            "system_metrics": metrics
        }

# Example usage:
if __name__ == "__main__":

    # Load your ground truth and agent result JSON records into dictionaries.
    # For example:
    # with open('ground_truth.json') as f:
    #     gts = json.load(f)
    # with open('agent_result.json') as f:
    #     agent_result = json.load(f)

    agent_metrics = AgentMetrics(gts, agent_result)
    overall_correctness, error_details = agent_metrics.correctness()
    metrics = agent_metrics.system_metrics()
    summary = agent_metrics.eval_all()

    print("Overall Correctness:", overall_correctness)
    print("Error Types:", error_details)
    print("System Metrics:", metrics)
    print("Full Summary:", summary)
