#!/usr/bin/env bash

rclient="openai"
rmodel="o4-mini"
rmodel="o3"

# Loop exp_id from 0 to 25 and run the Python script each time
for i in {0..21}; do
  # python3 main_single.py --exp_id "$i" --client "$rclient" --model "$rmodel" --flow_ver "$rmodel"
  python3 main_geoflow.py --exp_id "$i" --client "$rclient" --model "$rmodel" #--flow_ver "$rmodel"
  # python3 main_flow.py --exp_id "$i" --client "$rclient" --model "$rmodel" #--flow_ver "$rmodel"
  # python3 main_swarm.py --exp_id "$i" --client "$rclient" --model "$rmodel" #--flow_ver "$rmodel"
  # python3 main_seq_stateflow.py --exp_id "$i" --client "$rclient" --model "$rmodel" #--flow_ver "$rmodel"
  # python3 main_group_stateflow.py --exp_id "$i" --client "$rclient" --model "$rmodel" #--flow_ver "$rmodel"
done
