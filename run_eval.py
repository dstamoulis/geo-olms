from evaluate.agent_metrics import AgentMetrics

def main():

    gts_path = "results/openai/gpt_4o_mini/single_agent/"
    # results_path = "results/openai/gpt_4o_mini/geoflow/"
    # results_path = "results/openai/gpt_4o_mini/geoolm/"
    results_path = "results/openai/gpt_4o_mini/swarm/"
    results_path = "results/openai/gpt_4o_mini/swarm/"
    agent_metrics = AgentMetrics(gts_path, results_path)

    runs_from, runs_to = 0, 22
    runs_id = [i for i in range(runs_from, runs_to)]

    for run_id in runs_id:
        agent_metrics.evaluate_run(run_id)

    agent_metrics.print_avg_llm_metrics()
    agent_metrics.print_error_counts()
    agent_metrics.print_unsuccess_counts()
    agent_metrics.print_avg_agent_scores()

if __name__ == "__main__":
    main()