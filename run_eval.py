from evaluate.agent_metrics import AgentMetrics

def main():

    gts_file = './results/geeo25_minival.json' 
    agent_results_file =  './results/geeo25_minival.json' 

    gts_file = './results/geeo25_val_run2.json'
    agent_results_file = './results/geeo25_val_run1.json'

    agent_metrics = AgentMetrics(gts_file, agent_results_file)
    overall_correctness, error_details = agent_metrics.correctness()
    metrics = agent_metrics.system_metrics()
    # summary = agent_metrics.eval_all()

    print("Overall Correctness:", overall_correctness)
    print("Error Types:", error_details)
    print("System Metrics:", metrics)
    # print("Full Summary:", summary)

if __name__ == "__main__":
    main()