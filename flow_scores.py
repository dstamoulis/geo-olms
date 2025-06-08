import json
from collections import deque, defaultdict
from openai import OpenAI
api_client = OpenAI()

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

def evaluate_flow_correctness(gt_path, res_path):
    gt_js = json.load(open(gt_path))
    res_js = json.load(open(res_path))

    gt_tasks, gt_roots = load_graph(gt_js)
    res_tasks, res_roots = load_graph(res_js)

    error_types = defaultdict(int)
    num_gt = len(gt_tasks)
    denom = num_gt * 4 + 1

    # 0) step‐count error will be counted later if we didn't visit all GT nodes

    visited = set()
    queue = deque()

    # Initialize by matching each GT root to the first same‐agent res root
    for gr in gt_roots:
        gnode = gt_tasks[gr]
        for rr in res_roots:
            if res_tasks[rr]["agent"] == gnode["agent"]:
                queue.append((gr, rr))
                visited.add(gr)
                break

    # BFS over matched pairs
    while queue:
        gid, rid = queue.popleft()
        g, r = gt_tasks[gid], res_tasks[rid]

        # 1) objective
        if objective_score_llm(g["objective"], r["objective"]) < 4:
            error_types["wrong objective"] += 1
        # 2) agent
        if g["agent"] != r["agent"]:
            error_types["wrong agent"] += 1
        # 3) next count (structure)
        if set(g.get("next", [])) != set(r.get("next", [])):
            error_types["wrong next steps"] += 1
        # 4) prev count
        if set(g.get("prev", [])) != set(r.get("prev", [])):
            error_types["wrong prev steps"] += 1

        # Enqueue successors
        gsuccs = g.get("next", [])
        # build map agent→rid successor for quick lookup
        rsuccs = res_tasks[rid].get("next", [])
        agent_to_rsucc = {
            res_tasks[nid]["agent"]: nid for nid in rsuccs
        }
        for gn in gsuccs:
            if gn in visited:
                continue
            targ_agent = gt_tasks[gn]["agent"]
            rn = agent_to_rsucc.get(targ_agent)
            if rn:
                queue.append((gn, rn))
                visited.add(gn)
            else:
                # missing next
                error_types["wrong next steps"] += 1
                # we still mark visited so we don’t double‐count missing
                visited.add(gn)

    # 0) now that BFS is done, check total steps
    if len(visited) != num_gt:
        error_types["wrong number of steps"] += 1

    total_err = sum(error_types.values())
    score = max(0.0, 1.0 - total_err/denom)
    return score, error_types


if __name__ == "__main__":
    gt_path = 'prompt_tests/benchmark/geo_0/flow_gt.json'
    res_path = 'prompt_tests/benchmark/geo_0/flow_exp_2.json'
    score = evaluate_flow_correctness(gt_path, res_path)
    print(score)