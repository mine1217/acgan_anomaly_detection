#!/bin/sh
#前処理とかクラスタリングとかするsh

device_id=$1

#日付で訓練データと評価データを分割する時は引数を3ついれる　ランダム分割は2つでいい 論文はランダム分割でやる
if [ $# -eq 3 ]; then
    date=$2
    seed=$3
    python3 src/preprocess/make_train_dataset.py --input data/elect_data/train/$device_id.csv --output data/experiments/all/$device_id.csv
    python3 src/experiments/train_test_split.py --seed $seed --input data/experiments/all/$device_id.csv --date $date --train_save data/experiments/train/$device_id.csv --test data/experiments/test/$device_id.csv
    python3 src/preprocess/optimize.py --input data/experiments/train/$device_id.csv --save data/experiments/label/$device_id.csv --combination_save data/experiments/combination/$device_id.json --seed $seed
elif [ $# -eq 2 ]; then
    seed=$2
    python3 src/preprocess/make_train_dataset.py --input data/elect_data/train/$device_id.csv --output data/experiments/all/$device_id.csv
    python3 src/experiments/train_test_split.py --seed $seed --input data/experiments/all/$device_id.csv --train_save data/experiments/train/$device_id.csv --test data/experiments/test/$device_id.csv
    python3 src/preprocess/optimize.py --input data/experiments/train/$device_id.csv --save data/experiments/label/$device_id.csv --combination_save data/experiments/combination/$device_id.json --seed $seed
fi