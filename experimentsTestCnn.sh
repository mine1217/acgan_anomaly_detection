#!/bin/sh
if [ $# != 1 ] ; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTest.sh 503342
    exit 1
fi
device_id=$1
python3 src/experiments/cnn_test.py --input data/experiments/test/${device_id}.csv --model models/experiments/cnn/${device_id}.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --accuracy_save output/experiments/accuracy/${device_id}_cnn.csv --umap_save output/experiments/umap/cnn/${device_id} --roc_save output/experiments/roc_curve/${device_id}_cnn.png