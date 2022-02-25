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
現在停止中 動かないよ!!!!.

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

### 論文のデータの実験(ID5032B9)

- output/experiments/roc_curve/にroc曲線の図を出力．
- output/experiments/accuracy/に最適な閾値のときの精度をcsvファイルに出力．
- 5032AB_test_normal_w=0.1.csv，5032AB_test_anomaly_w=0.1.csvから再現可能．

```zsh
docker run -it --rm --name sensepro_anomaly_detection_evaluation -v $PWD:/workspace -w /workspace sensepro_anomaly_detection sh experimentsEvaluation.sh 5032B9
```

### 実験（一括)

以下のコードで学習から評価を一括でやってしまう 
複数回数学習～評価までを繰り返す デフォルトは10回
シェルスクリプトにデバイスIDと使用するモデル名を指定する 
- acgan
- cgan
- gan

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_preprocess -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsRoop.sh 5032B9 acgan
```

出力場所は以下の通り.
- モデル(重み)
  - models/experiments/ganモデル名/デバイス名_回数/*.h5
  - models/experiments/anoganモデル名/デバイス名_回数/*.h5
- lossの推移
  - output/experiments/loss/デバイス名_ganモデル名_回数.png
- 実験結果
  - output/experiments/score/デバイス名_anoganモデル名_回数_normal.csv
  - output/experiments/score/デバイス名_anoganモデル名_回数_anomaly.csv
  - output/experiments/roc_curve/デバイス名_ganモデル名_回数.png
  - output/experiments/accuracy/デバイス名_ganモデル名_回数.csv
- AnoGANで生成されたデータ
  - output/experiments/generated_data/デバイス名_anoganモデル名_回数_normal.csv
  - output/experiments/generated_data/デバイス名_anoganモデル名_回数_anomaly.csv
  - output/elect_graph/generated/デバイス名_anoganモデル名_回数_normal/*.png
  - output/elect_graph/generated/デバイス名_anoganモデル名_回数_anomaly/*.png
- 正常な検査データと生成データのユークリッド距離
  - output/experiments/euclidean/デバイス名_anoganモデル名_回数



(LSTM)GANを使用したい場合はモデルにはganを指定し,
experimentRoop.sh内の61行目の"src/acgan/gan.py"を実行する引数に"--l_gan"オプションを追加する.

```zsh
python3 src/acgan/${gan_model}.py 
  --l_gan
  --input data/experiments/train/$device_id.csv 
  --label data/experiments/label/$device_id.csv 
  --min_max_save data/experiments/minmax/$device_id.json 
  --model_save models/experiments/${gan_model}/${device_id}_${i}/ 
  --loss_save output/experiments/loss/${device_id}_${gan_model}_${i}.png 
```

### 実験（手順ごと)

以下 別々にやりたい場合(AC-GAN)

前処理・train，test分割
デバイスidとseed値を指定する

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_preprocess -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsPreprocess.sh 5032B9 0
```

train

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_train -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsTrain.sh 5032B9
```

test

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_test -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsTest.sh 5032B9
```

evaluation

- output/experiments/roc_curve/にroc曲線の図を出力．
- output/experiments/accuracy/に最適な閾値のときの精度をcsvファイルに出力．

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_evaluation -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsEvaluation.sh 503342
```

## Umap

以下のコードで論文に載せたUmapによる散布図を出力.
２種類のUmapを出力する.
- gmmでクラスタリングした結果を色分けして表示
  - output/experiments/umap/gmm
- 日付が進む毎に連続的に色を変えて表示
  - output/experiments/umap/weekday

```zsh
docker run --runtime=nvidia -it --rm --name sensepro_anomaly_detection_umap -v $PWD:/workspace -w /workspace minamotofordocker/sensepro_anomaly_detection 
  sh experimentsUmap.sh 5032B9
```

また、同時にクラスタ間距離をターミナルに表示する.
