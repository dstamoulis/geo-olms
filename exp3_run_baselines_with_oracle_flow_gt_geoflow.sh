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


# ------------------------------------------------------------------------
# Broken down to 0..21 bc some get stuck (comment runs that keep failing)

python3 main_geoflow.py --exp_id 0 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 1 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 2 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 3 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 4 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 5 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 6 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 7 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 8 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 9 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 10 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 11 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 12 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 13 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 14 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 15 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 16 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 17 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 18 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 19 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 20 --client "$rclient" --model "$rmodel"
python3 main_geoflow.py --exp_id 21 --client "$rclient" --model "$rmodel"
