#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsUmap.sh 5032AB
    exit 1
fi
device_id=$1
# gmmでクラスタリングした結果で色分けしたumap
python3 src/other/umap_ex.py --normalize --input data/experiments/all/$device_id.csv --combination data/experiments/combination/$device_id.json --save output/experiments/umap/gmm/$device_id.png
# 時間が経つ毎に連続的に色が変わるumap
python3 src/preprocess/umap_ex.py --normalize --weekday_label --input data/experiments/all/$device_id.csv --combination data/experiments/combination/$device_id.json --save output/experiments/umap/weekday/$device_id.png