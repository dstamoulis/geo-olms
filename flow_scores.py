import json
from collections import deque, defaultdict
from openai import OpenAI
api_client = OpenAI()


def group_tasks_by_agent_order(tasks, roots):
    visited = set()
    queue = deque(roots)
    agent_to_tasks = defaultdict(list)
    while queue:
        tid = queue.popleft()
        if tid in visited:
            continue
        visited.add(tid)
        task = tasks[tid]
        agent_to_tasks[task['agent']].append(task)
        queue.extend(task.get('next', []))
    return agent_to_tasks


def objective_score_llm(gt_objective: str, res_objective: str) -> int:
    """
    Uses GPT-4o-mini to rate how well res_objective matches gt_objective
    on a 1–5 scale, where 5 means “fully captures dataset, dates, location,
    plot type, model/detection, etc.” and 1 means “none of the required info.”
    """
    prompt = f"""
    You are an evaluator. Compare these two task objectives:

    Ground truth objective:
    \"\"\"
    {gt_objective}
    \"\"\"

    Result objective:
    \"\"\"
    {res_objective}
    \"\"\"

    Rate on an integer scale from 1 to 5 how well the result objective captures what the ground-truth. 
    
    IMPORTANT: WE DO NOT WANT EXACT TEXTUAL MATCH. WHAT WE WANT IS THAT THE FOLLOWING DETAILS 
    -- ONLY IF PRESENT IN THE GT OBJECTIVE -- THEY ARE SUFFICIENTLY SPELLED OUT IN THE RESULT DETAILS!!

    --

    That is, we don't want exact match but if the ground truth for example has some of the following
    - dataset name  (if applicable) -- AND/OR ...
    - date or date range  (if applicable) -- AND/OR ...
    - geographical location  (if applicable) -- AND/OR ...
    - plot or processing type  (if applicable) -- AND/OR ...
    - detection/model details (if applicable) ...
    Then check that are present in the results objective!

    Respond with exactly the integer score (1, 2, 3, 4, or 5), no other text!!
    """
    # print(gt_objective, res_objective)

    resp = api_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rate task objectives for completeness."},
            {"role": "user", "content": prompt}
        ]
    )
    # The assistant's content should be the integer
    score_str = resp.choices[0].message.content.strip()
    # print(score_str)
    try:
        return int(score_str)
    except ValueError:
        # fallback if parsing fails
        return 1


def load_graph(js):
    """Return a dict of nodes and a root list (ids with empty prev)."""
    tasks = js["tasks"]
    roots = [tid for tid, t in tasks.items() if not t.get("prev")]
    return tasks, roots

def match_tasks(gt_group, res_group, error_types):
    matched = set()
    for gt_task in gt_group:
        best_score = -1
        best_res = None
        for res_task in res_group:
            if res_task['id'] in matched:
                continue
            score = objective_score_llm(gt_task["objective"], res_task["objective"])
            if score > best_score:
                best_score = score
                best_res = res_task

        if best_res is None:
            error_types["missing task"] += 1
            continue

        matched.add(best_res['id'])

        if best_score < 4:
            error_types["wrong objective"] += 1
        if gt_task["agent"] != best_res["agent"]:
            error_types["wrong agent"] += 1
        if set(gt_task.get("next", [])) != set(best_res.get("next", [])):
            error_types["wrong next steps"] += 1
        if set(gt_task.get("prev", [])) != set(best_res.get("prev", [])):
            error_types["wrong prev steps"] += 1

def evaluate_flow_correctness(gt_path, res_path):
    gt_js = json.load(open(gt_path))
    res_js = json.load(open(res_path))

    gt_tasks, gt_roots = load_graph(gt_js)
    res_tasks, res_roots = load_graph(res_js)

    # Group by agent and order
    gt_agent_groups = group_tasks_by_agent_order(gt_tasks, gt_roots)
    res_agent_groups = group_tasks_by_agent_order(res_tasks, res_roots)

    error_types = defaultdict(int)
    num_gt = len(gt_tasks)
    denom = num_gt * 4 + 1

    for agent in gt_agent_groups:
        gt_group = gt_agent_groups[agent]
        res_group = res_agent_groups.get(agent, [])
        match_tasks(gt_group, res_group, error_types)

    # Additional check: total step count
    if len(gt_tasks) != len(res_tasks):
        error_types["wrong number of steps"] += 1

    total_err = sum(error_types.values())
    score = max(0.0, 1.0 - total_err / denom)
    return score, error_types


if __name__ == "__main__":
    for i in range(22):
        gt_path = f'prompt_tests/benchmark/geo_{i}/flow_gt.json'
        res_path = f'prompt_tests/benchmark/geo_{i}/flow_exp_2.json'
        score = evaluate_flow_correctness(gt_path, res_path)
        print(f'Exp {i}: {score}')