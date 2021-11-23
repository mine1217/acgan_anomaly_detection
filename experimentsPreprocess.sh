#!/bin/sh
#files="./data/elect_data/train/*"
#device_id=$(basename $filepath .csv)
if [ $# != 2 ]; then
    echo Please specify device_id,date in the argument.
    echo Example:sh experimentsPreprocess.sh 503342 2020.01.01
    exit 1
fi
device_id=$1
date=$2

python3 src/preprocess/make_train_dataset.py --input data/elect_data/train/$device_id.csv --output data/experiments/all/$device_id.csv
python3 src/experiments/train_test_split.py --input data/experiments/all/$device_id.csv --date $date --train_save data/experiments/train/$device_id.csv --test data/experiments/test/$device_id.csv
python3 src/preprocess/optimize.py --input data/experiments/train/$device_id.csv --save data/experiments/label/$device_id.csv --combination_save data/experiments/combination/$device_id.json