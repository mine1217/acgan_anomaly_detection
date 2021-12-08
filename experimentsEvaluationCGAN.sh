#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsEvaluation.sh 503342
    exit 1
fi
device_id=$1
python3 src/experiments/evaluation.py --normal output/experiments/score/canogan/${device_id}_normal.csv --anomaly output/experiments/score/canogan/${device_id}_anomaly.csv --roc_save output/experiments/roc_curve/${device_id}_CGAN.png --accuracy_save output/experiments/accuracy/${device_id}_CGAN.csv