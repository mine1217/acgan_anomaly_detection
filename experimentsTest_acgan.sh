#!/bin/sh
if [ $# != 1 ]; then
    echo Please specify device_id in the argument.
    echo Example:sh experimentsTest.sh 503342
    exit 1
fi
device_id=$1
python3 src/experiments/acanogan_test.py --input data/experiments/test/${device_id}.csv --g_model models/acgan/${device_id}/generator.h5 --d_model models/acgan/${device_id}/discriminator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/acanogan/${device_id}_normal.csv --model_save output/experiments/acanogan/models/${device_id}_normal/
python3 src/experiments/acanogan_test.py --is_anomaly_test --input data/experiments/test/${device_id}.csv --g_model models/acgan/${device_id}/generator.h5 --d_model models/acgan/${device_id}/discriminator.h5 --combination data/experiments/combination/${device_id}.json --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/acanogan/${device_id}_anomaly.csv --model_save output/experiments/acanogan/models/${device_id}_anomaly/