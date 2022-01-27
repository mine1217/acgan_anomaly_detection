#!/bin/sh
device_id=$1
num=$2
model=$3

python3 src/other/c_distance.py --input data/experiments/all/${device_id}.csv --combination data/experiments/combination/$device_id.json --save output/experiments/euclidean/${model}/${device_id}_${num}.csv
