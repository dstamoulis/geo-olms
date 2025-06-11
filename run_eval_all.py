from evaluate.agent_metrics import AgentMetrics
import os
from utils import re_args_component

def main():

    gts_path = "results/flow_gt/openai/gpt_4o_mini/single_agent/"

    run_client = "openai" # openai, ollama
    run_model = "gpt-4o-mini"
    run_flow_ver = run_model #"flow_gt"
    run_flow_ver = "flow_gt"

    run_methods = ['single_agent', 'geoflow', 'geoolm', 'swarm', 'seq_stateflow', 'group_stateflow']
    run_methods = ['geoflow', 'geoolm', 'swarm', 'seq_stateflow', 'group_stateflow']
    run_methods = ['geoolm', 'geoflow']

    for run_method in run_methods:

        results_path = f"results/{re_args_component(run_flow_ver)}/{re_args_component(run_client)}/{re_args_component(run_model)}/{re_args_component(run_method)}/"
        agent_metrics = AgentMetrics(gts_path, results_path)

        runs_from, runs_to = 0, 22
        runs_id = [i for i in range(runs_from, runs_to)]

        for run_id in runs_id: 
            agent_metrics.evaluate_run(run_id)

        print(f"Run: {run_method}")
        agent_metrics.print_avg_llm_metrics()
        # agent_metrics.print_error_counts()
        # agent_metrics.print_unsuccess_counts()
        agent_metrics.print_avg_agent_scores()
        agent_metrics.missing_runs_stats()

if __name__ == "__main__":
    main()