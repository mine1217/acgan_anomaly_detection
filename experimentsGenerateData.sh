#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTest.sh 503342
    exit 1
fi
device_id=$1
python3 src/experiments/generate.py --g_model models/experiments/acgan/${device_id}/generator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --save output/experiments/generated_data/${device_id}_noise.csv
#python3 src/experiments/generateR.py --g_model models/experiments/racgan/${device_id}/generator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --save output/experiments/generated_data/${device_id}_noise.csv
#python3 src/experiments/generateR.py --g_model models/experiments/rcgan/${device_id}/generator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --save output/experiments/generated_data/${device_id}_noise.csv
python3 src/experiments/plot_data.py --input output/experiments/generated_data/${device_id}_noise.csv --save output/elect_graph/generated/${device_id}_noise/