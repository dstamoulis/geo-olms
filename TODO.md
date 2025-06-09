## TODOs

* @Stanley: geoolm++ is coded using ResponsesAPI (without having a ChatCompl options) so incompatible is ollama => add ChatCompletions OpenAI support (focus on OpenAI only; ollama follows exactly the same syntax, so you won't need to touch ollama code)
* @Justing: Line 252 assistant.py: errors out when the response is anything BUT the exact match you are expecting ==> add parser to handle cases (e.g., if the LLM reasons or think but correctly says NEXT or DONE in the end, then make sure to parse that last part and proceed correctly)
* Add back gpt_prompt_ollama @Janani-Dimi
* Add Token-runtime cost in agent_eval @Dimi
* Compile list of experiments @Dimi-Janani
* Update flow-correctness score @Stanley
* Paper skeleton @Justin


## Experiments

### exp1_generate_flows_ollama.sh

Generate flows with ollama models:
* [x] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### exp2_run_baselines_with_oracle_flow_gt_single.sh

Single: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### exp3_run_baselines_with_oracle_flow_gt_geoflow.sh

GeoFlow: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### exp4_run_baselines_with_oracle_flow_gt_swarm.sh

Swarm: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### exp5_run_baselines_with_oracle_flow_gt_seq_stateflow.sh

Seq-StateFlow: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### [SKIP! WIP] exp6_run_baselines_with_oracle_flow_gt_group_stateflow.sh

Group-StateFlow: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b


### [SKIP! WIP] exp7_run_baselines_with_oracle_flow_gt_geoolm.sh

Geo-OLM++: Run each baseline with different ollama models with oracle `flow_gt` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b



### exp8_run_baselines_with_ollama_flows_geoflow.sh

GeoFlow: Run each baseline with different ollama models with their respective ollama-generated `flows` as input:

* [ ] qwen3:1.7b
* [ ] qwen3:4b
* [ ] qwen3:8b
* [ ] qwen3:14b
* [ ] qwen3:32b
* [ ] llama3.3:70b
* [ ] phi4-mini:3.8b
* [ ] mistral-small:24b
