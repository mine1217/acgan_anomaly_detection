# coding: UTF-8
# Numpy
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib
import datetime
import jpholiday
import json

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
parser.add_argument(
    "-c",
    "--combination",
    default="data/experiments/combination/5032AB.json",
    help="combination(best label to date) file path")
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
ylim = max(40, ylim)

def encode_day_to_label(df: pd.DataFrame) -> dict:
    """
    日付を曜日，祝日により数値にエンコードする．[月,日]=[0,6]，祝日=7

    Args:
        df(pandas DataFrame):Daily data
    Returns:
        dict:{date\:label}
    """
    index_list = df.index.to_list()
    dates = list(map(lambda x: datetime.datetime.strptime(
        "20" + x, '%Y.%m.%d'), index_list))
    label = list(map(lambda x: x.weekday(), dates))
    holiday_label = list(map(lambda x: jpholiday.is_holiday(
        datetime.date(x.year, x.month, x.day)), dates))
    label = [7 if holiday_label[i] else label[i] for i in range(len(label))]
    return dict(zip(index_list, label))

def mkdir(dir):
    dir_path = os.path.dirname(dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

combination = json.load(open(args.combination))
combination = list(combination.values())
input = pd.read_csv(args.input, header=None, index_col=0)
day_to_label = encode_day_to_label(input).values()
class_labels = [combination[i] for i in day_to_label]
num_classes = int(max(combination)) + 1

x = input.values
print(class_labels)

for n in range(num_classes):
    data = x[class_labels==n]
    print(data)
    x = np.linspace(0, 24, 120)  # x軸データ(0,1,2...,24)
    axes = []  # 各データのグラフ
    mkdir(args.save + "class" + str(n) + "/")
    plt.xlabel("Hour")
    plt.ylabel("Power")
    # plt.xticks(x, [0,6,12,18,24])
    plt.yticks(color="None")
    for i, data in enumerate(data):
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
        plt.savefig(args.save+ "class" + str(n) + days[i] + ".png")  # 保存
        plt.close()
