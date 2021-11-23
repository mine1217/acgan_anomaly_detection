#!/bin/sh
python2.7 src/preprocess/make_daily_data.py
python3 src/preprocess/detection_setting.py

files="./data/processed/detect/*"
for filepath in $files; do
    device_id=$(basename $filepath .csv)
    python3 src/acanogan/detect_anomaly.py --input data/elect_data/detect/$device_id.csv --g_model models/acgan/$device_id/generator.h5 --d_model models/acgan/$device_id/discriminator.h5 --combination data/processed/combination/$device_id.json --minmax data/processed/minmax/$device_id.json --output output/deviceState/$device_id.json
done
