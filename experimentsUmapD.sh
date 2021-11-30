#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrain.sh 503342
    exit 1
fi
device_id=$1
python3 src/preprocess/umap_d.py --input data/experiments/test/$device_id.csv --d_model models/experiments/acgan/${device_id}/discriminator.h5 --minmax data/experiments/minmax/${device_id}.json --combination data/experiments/combination/$device_id.json --save output/experiments/umap/discriminator/$device_id.png