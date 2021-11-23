# sensepro_anomaly_detection

Anomaly detection for sensepro life watching system．

プロジェクト全体の資料等はGoogle Driveの[lab/プロジェクト/センスプロ](https://drive.google.com/drive/u/0/folders/1ZDGCE8IGZrwjf0DMM8UYymZv4dlZ9dM1).

## Environment

### 運用設定

docker imageがなければimage fileからload．

```zsh
cd /home/ci/sensepro_anomaly_detection
docker load < sensepro_anomaly_detection.tar
```

ci@shimuraのcronで下記を設定して定期実行している．

```cron
10 4 1 * * (cd /home/ci/sensepro_anomaly_detection; sh train.sh)
10 0 * * * (cd /home/ci/sensepro_anomaly_detection; sh detect.sh)
```

- 現状はサーバ(ci@shimura)で定期実行し，/home/ci/sensepro_anomaly_detection/output/deviceState/下のjsonファイルに各デバイスの状態(stable・unstable)を書き込んでいる．
- データ数が30日以下のものと1日の間に一度も利用されていない日の場合は，とりあえずstableとしている．
- jsonファイルはci@ci11.info.kindai.ac.jp:/var/www/html/sp/deviceState/に送信される．
- アプリはci@ci11.info.kindai.ac.jp:/var/www/html/sp/deviceState/のjsonファイルを参照する．
- /home/ci/sensepro_anomaly_detectionは本番環境となるので，ここで実験を行わないこと．

### 実験設定

最新データのダウンロードとデータセットの作成

```zsh
rsync -u ci@ci11.info.kindai.ac.jp:/var/www/html/sp/log*.csv data/raw
python2.7 src/preprocess/make_daily_data.py --train
```

## Experiments

### 論文のデータの実験(ID5032AB)

- output/experiments/roc_curve/にroc曲線の図を出力．
- output/experiments/accuracy/に最適な閾値のときの精度をcsvファイルに出力．
- 5032AB_test_normal_w=0.1.csv，5032AB_test_anomaly_w=0.1.csvから再現可能．

```zsh
docker run -it --rm --name sensepro_anomaly_detection_evaluation -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh experimentsEvaluation.sh 5032AB
```

### その他のデバイス(ID503342の例)

ある院生のモニターの2020.01.01以降のデータを検証用データとして実験を行う例．

前処理・train，test分割

```zsh
docker run -it --rm --name sensepro_anomaly_detection_preprocess -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh experimentsPreprocess.sh 503342 2020.01.01 
```

train

```zsh
docker run -it --rm --name sensepro_anomaly_detection_train -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh experimentsTrain.sh 503342
```

test

```zsh
docker run -it --rm --name sensepro_anomaly_detection_test -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh experimentsTest.sh 503342
```

evaluation

- output/experiments/roc_curve/にroc曲線の図を出力．
- output/experiments/accuracy/に最適な閾値のときの精度をcsvファイルに出力．

```zsh
docker run -it --rm --name sensepro_anomaly_detection_evaluation -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection sh experimentsEvaluation.sh 503342
```

## Docs

ブラウザから
[docs/\_build/index.html](docs/_build/index.html)
を参照．
