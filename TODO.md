
* @Stanley: geoolm++ is coded using ResponsesAPI (without having a ChatCompl options) so incompatible is ollama => add ChatCompletions OpenAI support (focus on OpenAI only; ollama follows exactly the same syntax, so you won't need to touch ollama code)
* @Justing: Line 252 assistant.py: errors out when the response is anything BUT the exact match you are expecting ==> add parser to handle cases (e.g., if the LLM reasons or think but correctly says NEXT or DONE in the end, then make sure to parse that last part and proceed correctly)


----

* Add back gpt_prompt_ollama @Janani-Dimi
* Add Token-runtime cost in agent_eval @Dimi
* Compile list of experiments @Dimi-Janani
* Update flow-correctness score @Stanley
* Paper skeleton @Justin

-----

