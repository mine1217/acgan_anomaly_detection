#!/bin/sh
rsync -u ci@ci11.info.kindai.ac.jp:/var/www/html/sp/log*.csv data/raw
docker run -it --rm --name sensepro_anomaly_detection_production_train -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh trainPyProcess.sh