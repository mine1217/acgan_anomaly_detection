#!/bin/sh

# 学習から評価まで一括で行うsh 複数回数実行できる デフォルトは10回

# device_id 家電番号, model 使用するモデル名, num 実行回数
device_id="5032AB"
model="acgan"
num=10

#引数が3つ与えられた時は実行回数も指定できる
if [ $# = 2 ]; then
    device_id=$1
elif [ $# = 3 ]; then
    device_id=$1
    num=$3
else
    echo Please specify device_id in the argument.
    echo Example1:sh experimentsTrain.sh 5032AB acgan
    echo Example2:sh experimentsTrain.sh 5032AB acgan 10
    exit 1
fi

#使用するモデル名
ACGAN="acgan"
ACANOGAN="acanogan"
CGAN="cgan"
CANOGAN="canogan"
GAN="gan"
ANOGAN="anogan"

case "$2" in
  ${ACGAN})
    gan_model=$ACGAN
    anogan_model=$ACANOGAN
    ;;
  ${CGAN})
    gan_model=$CGAN
    anogan_model=$CANOGAN
    ;;
  ${GAN})
    gan_model=$GAN
    anogan_model=$ANOGAN
    ;;
  *)
    echo "not found a model such name $2"
    ;;
esac

#前の実験データを一括で削除
# rm -rf output/experiments/score/${device_id}*${anogan_model}*
# rm -rf output/experiments/roc_curve/${device_id}*${anogan_model}*
# rm -rf output/experiments/accuracy/${device_id}*${anogan_model}*
# rm -rf output/elect_graph/generated/${device_id}*${anogan_model}*
# rm -rf models/experiments/${gan_model}/${device_id}*
# rm -rf models/experiments/${anogan_model}/${device_id}*

#num回数　学習~評価繰り返し
for i in `seq $num`
do
    echo "*************************実行 $i 回目*************************"
    python3 src/acgan/${gan_model}.py --input data/experiments/train/$device_id.csv --label data/experiments/label/$device_id.csv --min_max_save data/experiments/minmax/$device_id.json --model_save models/experiments/${gan_model}/${device_id}_${i}/ --loss_save output/experiments/loss/${device_id}_${gan_model}_${i}.png 
    sleep 10s
    python3 src/experiments/${anogan_model}_test.py --input data/experiments/test/${device_id}.csv --combination data/experiments/combination/${device_id}.json --model models/experiments/${gan_model}/${device_id}_${i}/ --minmax data/experiments/minmax/${device_id}.json --score_save output/experiments/score/${device_id}_${anogan_model}_${i}_normal.csv --model_save models/experiments/${anogan_model}/ --generated_save output/experiments/generated_data/${device_id}_${anogan_model}_${i}_normal.csv
    sleep 10s
    python3 src/experiments//${anogan_model}_test.py --is_anomaly_test --input data/experiments/test/${device_id}.csv --combination data/experiments/combination/${device_id}.json --model models/experiments/${gan_model}/${device_id}_${i}/ --minmax data/experiments/minmax/${device_id}.json --model_save models/experiments/${anogan_model}/ --score_save output/experiments/score/${device_id}_${anogan_model}_${i}_anomaly.csv --generated_save output/experiments/generated_data/${device_id}_${anogan_model}_${i}_anomaly.csv
    sleep 10s
    python3 src/experiments/evaluation.py --normal output/experiments/score/${device_id}_${anogan_model}_${i}_normal.csv --anomaly output/experiments/score/${device_id}_${anogan_model}_${i}_anomaly.csv --roc_save output/experiments/roc_curve/${device_id}_${anogan_model}_${i}.png --accuracy_save output/experiments/accuracy/${device_id}_${anogan_model}_${i}.csv
    python3 src/experiments/plot_data.py --input output/experiments/generated_data/${device_id}_${anogan_model}_${i}_normal.csv --save output/elect_graph/generated/acanogan/${device_id}_${anogan_model}_${i}_normal/
    python3 src/experiments/plot_data.py --input output/experiments/generated_data/${device_id}_${anogan_model}_${i}_anomaly.csv --save output/elect_graph/generated/acanogan/${device_id}_${anogan_model}_${i}_anomaly/
    python3 src/other/euclidean.py --combination data/experiments/combination/${device_id}.json --inputA data/experiments/test/${device_id}.csv --inputB output/experiments/generated_data/${device_id}_${anogan_model}_${i}_normal.csv --save output/experiments/euclidean/acanogan/${device_id}_${anogan_model}_${i}.csv
done