#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTrain.sh 503342
    exit 1
fi
device_id=$1

for i in 1 2 3 4 5 6 7 8 9 10
do
    python3 src/acgan/cgan.py --input data/experiments/train/$device_id.csv --label data/experiments/label/$device_id.csv --min_max_save data/experiments/minmax/$device_id.json --model_save models/experiments/cgan/${device_id}_${i}/ --loss_save output/experiments/cgan_loss/${device_id}_${i}.png 
    python3 src/experiments/canogan_test.py --input data/experiments/test/${device_id}.csv --g_model models/experiments/cgan/${device_id}_${i}/generator.h5 --d_model models/experiments/cgan/${device_id}_${i}/discriminator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/canogan/${device_id}_${i}_normal.csv --model_save output/experiments/canogan/models/${device_id}_${i}_normal/ --generated_save output/experiments/generated_data/canogan/${device_id}_${i}_normal.csv
    python3 src/experiments/canogan_test.py --is_anomaly_test --input data/experiments/test/${device_id}.csv --g_model models/experiments/cgan/${device_id}_${i}/generator.h5 --d_model models/experiments/cgan/${device_id}_${i}/discriminator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/canogan/${device_id}_${i}_anomaly.csv --model_save output/experiments/canogan/models/${device_id}_${i}_anomaly/ --generated_save output/experiments/generated_data/canogan/${device_id}_${i}_anomaly.csv
    python3 src/experiments/evaluation.py --normal output/experiments/score/canogan/${device_id}_${i}_normal.csv --anomaly output/experiments/score/canogan/${device_id}_${i}_anomaly.csv --roc_save output/experiments/roc_curve/${device_id}_${i}.png --accuracy_save output/experiments/accuracy/${device_id}_${i}.csv
    python3 src/experiments/plot_data.py --input output/experiments/generated_data/canogan/${device_id}_${i}_normal.csv --save output/elect_graph/generated/${device_id}_${i}_normal/
    python3 src/experiments/plot_data.py --input output/experiments/generated_data/canogan/${device_id}_${i}_anomaly.csv --save output/elect_graph/generated/${device_id}_${i}_anomaly/
done