#!/usr/bin/env bash

# Source and destination base directories
SRC_BASE="prompt_tests/benchmark"
DST_BASE="prompt_flows/flows/flow_gpt_4o_mini"

# Ensure destination directory exists
mkdir -p "$DST_BASE"

# Copy each flow_gt.json as N.json
for i in {0..21}; do
  SRC="$SRC_BASE/geo_$i/flow_exp_2.json"
  DST="$DST_BASE/$i.json"

  if [ -f "$SRC" ]; then
    cp "$SRC" "$DST"
    echo "Copied geo_$i → $DST"
  else
    echo "WARNING: $SRC not found, skipping."
  fi
done
