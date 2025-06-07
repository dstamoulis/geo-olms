#!/usr/bin/env bash

# Loop exp_id from 0 to 25 and run the Python script each time
for i in {2..6}; do
  python3 main_single.py --exp_id "$i"
  python3 main_geoflow.py --exp_id "$i"
done
