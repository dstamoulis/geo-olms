#!/usr/bin/env bash

rclient="ollama"
# List of ollama models
#
rmodel="qwen3:1.7b" # DONE? [x]
rmodel="qwen3:4b" # DONE? [x]
rmodel="qwen3:8b" # DONE? [x]
rmodel="qwen3:14b" # DONE? [x]
rmodel="qwen3:32b" # DONE? [x]
#
rmodel="llama3.3:70b" # DONE? [ ]
#
rmodel="phi4-mini:3.8b" # DONE? [ ]
#
rmodel="mistral-small:24b" # DONE? [ ]

for i in {0..21}; do
    python3 prompt_flow_gen.py --exp_id "$i" --client "$rclient" --model "$rmodel"
done