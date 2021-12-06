#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrain.sh 503342
    exit 1
fi
device_id=$1
python3 src/experiments/plot_data_class.py --input data/experiments/all/$device_id.csv --combination data/experiments/combination/${device_id}.json --save output/experiments/elect_graph/$device_id/
python3 src/experiments/plot_data.py --input output/experiments/generated_data/${device_id}_normal.csv --save output/elect_graph/generated/${device_id}_normal/
python3 src/experiments/plot_data.py --input output/experiments/generated_data/${device_id}_anomaly.csv --save output/elect_graph/generated/${device_id}_anomaly/
