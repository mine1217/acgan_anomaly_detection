#!/bin/sh
python2.7 src/preprocess/make_daily_data.py --train

files="./data/elect_data/train/*"
for filepath in $files; do
    # echo $(basename $filepath .csv)
    device_id=$(basename $filepath .csv)
    python3 src/preprocess/make_train_dataset.py --input data/elect_data/train/$device_id.csv --output data/processed/train/$device_id.csv
done

files="./data/processed/train/*"
for filepath in $files; do
    device_id=$(basename $filepath .csv)
    python3 src/preprocess/optimize.py --input data/processed/train/$device_id.csv --save data/processed/label/$device_id.csv --combination_save data/processed/combination/$device_id.json
    python3 src/acgan/acgan.py --input data/processed/train/$device_id.csv --label data/processed/label/$device_id.csv --min_max_save data/processed/minmax/$device_id.json --model_save models/acgan/$device_id/
done
