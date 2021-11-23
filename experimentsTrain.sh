#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrain.sh 503342
    exit 1
fi
device_id=$1
python3 src/acgan/acgan.py --input data/experiments/train/$device_id.csv --label data/experiments/label/$device_id.csv --min_max_save data/experiments/minmax/$device_id.json --model_save models/experiments/acgan/$device_id/ --loss_save output/experiments/acgan_loss/$device_id.png 
