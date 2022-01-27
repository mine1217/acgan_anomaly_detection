#!/bin/sh
device_id=$1
num=$2
model=$3

python3 src/other/euclidean.py --combination data/experiments/combination/${device_id}.json --inputA data/experiments/test/${device_id}.csv --inputB output/experiments/generated_data/${model}/${device_id}_${num}_normal.csv --save output/experiments/euclidean/${model}/${device_id}_${num}.csv
