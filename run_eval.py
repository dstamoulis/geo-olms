from evaluate.agent_metrics import AgentMetrics

def main():

    gts_prefix = "results/openai/gpt_4o_mini/single_agent/openai_gpt_4o_mini_0_1_single_agent"
    results_prefix = "results/openai/gpt_4o_mini/geoflow/openai_gpt_4o_mini_0_1_geoflow"
    agent_metrics = AgentMetrics(gts_prefix, results_prefix)

    runs_from, runs_to = 0, 6
    runs_id = [i for i in range(runs_from, runs_to)]

    for run_id in runs_id:
        gts_file = f"{gts_prefix}_{run_id}.json"
        results_file = f"{results_prefix}_{run_id}.json"
        agent_metrics.evaluate_run(gts_file, results_file)

    agent_metrics.print_avg_llm_metrics()
    agent_metrics.print_error_counts()
    agent_metrics.print_overall_correctness()


if __name__ == "__main__":
    main()