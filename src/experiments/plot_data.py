# coding: UTF-8
# Numpy
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib

matplotlib.use("Agg")  # グラフ保存
'''
複数の時系列データ(電力データ)の波形のグラフを保存
'''
# 引数設定
parser = argparse.ArgumentParser(
    description='')
parser.add_argument(
    '--input',
    '-i',
    default="data/experiments/train/5032AB.csv",
    type=str,
    help='入力データ，表示する時系列データファイル')
parser.add_argument(
    '--save',
    '-s',
    default="output/elect_graph/5032AB/",
    type=str,
    help='保存先ディレクトリ')
args = parser.parse_args()  # 引数

days = []
input_data = []
ylim = 0  # y軸表示最大値，データの中でもっとも大きい値
# 引数のinput_dataのcsvをリストとして読み込む
csv_file = csv.reader(open(args.input),
                      delimiter=",", lineterminator="\r\n")
for line in csv_file:
    days.append(line[0])
    input = list(map(lambda str: float(str), line[1:]))  # 文字列をfloat変換
    input_data.append(input)
    day_max = max(input)
    if ylim < day_max:
        ylim = day_max
        print(line[0])
        print(day_max)
ylim = max(128, ylim)


def mkdir(dir):
    dir_path = os.path.dirname(dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


x = np.linspace(0, 24, 120)  # x軸データ(0,1,2...,24)
axes = []  # 各データのグラフ
mkdir(args.save)
plt.xlabel("Hour")
plt.ylabel("Power")
# plt.xticks(x, [0,6,12,18,24])
plt.yticks(color="None")
for i, data in enumerate(input_data):
    plt.rcParams["font.size"] = 25
    fig = plt.figure(figsize=(8, 8))  # figureオブジェクト作成
    plt.title(days[i])
    # plt.yticks(color="None")
    plt.tick_params(length=0)
    y = data  # y軸データ
    plt.xlabel("Hour")
    plt.ylabel("Power")
    plt.xticks([0, 6, 12, 18, 24])
    plt.xlim([0, 24])
    plt.ylim([0, ylim])
    # print("test")
    plt.tight_layout()
    plt.plot(x, y, "-o", lw=7)  # 波形グラフ作成，線の太さ変更
    # plt.plot(x, y)  # グラフ作成
    plt.savefig(args.save + days[i] + ".png")  # 保存
    plt.close()
