#!/bin/sh
# CNNを学習
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrainCnn.sh 503342
    exit 1
fi
device_id=$1
python3 src/acgan/cnnClassfier.py --input data/experiments/train/$device_id.csv --label data/experiments/label/$device_id.csv --min_max_save data/experiments/minmax/$device_id.json --model_save models/experiments/cnn/$device_id --loss_save output/experiments/cnn_loss/$device_id.png 
