#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrain.sh 503342
    exit 1
fi
device_id=$1
# python3 src/experiments/anogan_test.py --input data/experiments/test/${device_id}.csv --model models/experiments/gan/${device_id}/ --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/anogan/${device_id}_normal.csv --model_save output/experiments/anogan/models/${device_id}_normal/ --generated_save output/experiments/generated_data/${device_id}_normal.csv
# python3 src/experiments/anogan_test.py --is_anomaly_test --input data/experiments/test/${device_id}.csv --model models/experiments/gan/${device_id}/ --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/anogan/${device_id}_anomaly.csv --model_save output/experiments/anogan/models/${device_id}_anomaly/ --generated_save output/experiments/generated_data/${device_id}_anomaly.csv
python3 src/experiments/evaluation.py --normal output/experiments/score/anogan/${device_id}_normal.csv --anomaly output/experiments/score/anogan/${device_id}_anomaly.csv --roc_save output/experiments/roc_curve/${device_id}_ensemble.png --accuracy_save output/experiments/accuracy/${device_id}_ensemble.csv