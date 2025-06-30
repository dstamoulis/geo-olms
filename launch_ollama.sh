#!/usr/bin/env bash

rclient="ollama"
rmodel="qwen3:8b"

for i in {0..0}; do
  python3 main_single.py --exp_id "$i" --client "$rclient" --model "$rmodel"
  # python3 main_geoflow.py --exp_id "$i" --client "$rclient" --model "$rmodel"
  # python3 main_flow.py --exp_id "$i" --client "$rclient" --model "$rmodel"
  # python3 main_swarm.py --exp_id "$i"  --client "$rclient" --model "$rmodel"
done



