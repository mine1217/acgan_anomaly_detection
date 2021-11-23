#!/bin/sh
rsync -u ci@ci11.info.kindai.ac.jp:/var/www/html/sp/log*.csv data/raw/
rsync -u ci@ci11.info.kindai.ac.jp:/var/www/html/sp/deviceList.json data/raw/
docker run -it --rm --name sensepro_anomaly_detection_production_detect -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh detectPyProcess.sh
rsync -u output/deviceState/* ci@ci11.info.kindai.ac.jp:/var/www/html/sp/deviceState/
