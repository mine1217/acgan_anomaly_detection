"""
クラスタ間の距離を求める．
群平均法を使用した場合と、クラスタの重心間の距離を使用した場合の２つの方法で距離を求める. umapは使用しない.
umapで次元圧縮してから距離を求めるのは、umap_exでやる 卒論で出した距離はこのコードではなくumap_ex

Example:
    5032AB example
    ::
        python3 src/other/c_distance.py 
            -iA data/experiments/test/5032AB.csv"
            -c data/experiments/combination/5032AB.json 
            -s output/experiments/euclidean/5032AB_acgan_1.csv
"""
import _pathmagic
import random
import argparse
import json
import os

import numpy as np
import pandas as pd
import datetime
import jpholiday

from matplotlib import pyplot as plt
import umap.umap_ as umap


def normalize(x: np.array) -> tuple:
    """
    Min max normalize．

    Args:
        x:input data
    Returns:
        x_train, minimum, maximum
    """
    minimum = x.min(axis=None)
    maximum = x.max(axis=None)
    return (x - minimum) / (maximum - minimum), (minimum, maximum)

def main():
    args = arg_parse()

    #予測に必要なデータロード
    input = pd.read_csv(args.input, header=None, index_col=0)

    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = encode_day_to_label(input).values()
    class_labels = [combination[i] for i in day_to_label]
    num_classes = int(max(combination)) + 1

    #Encode label
    label = encode_day_to_label(input)

    x, y = input.values, class_labels

    x, (minimum, maximum) = normalize(x)


    distance = np.zeros((num_classes , num_classes))
    distanceCG = np.zeros((num_classes , num_classes))
    var = np.zeros((num_classes, 120))
    sumAll = 0
    sumAllCG = 0
    for i in np.unique(y):
        dataA = x[y == i]
        for n in np.unique(y):
            if n == i: 
                continue
            dataB = x[y == n]
            avg = 0
            sum = 0
            for a in dataA:
                for b in dataB:
                    sum += np.linalg.norm(a-b)
            avg = sum / (len(dataA) * len(dataB))
            sumAll += avg
            distance[i][n] = np.round(avg, decimals=4)
            distanceCG[i][n] = np.round(np.linalg.norm(np.average(dataA, axis = 0) - np.average(dataB, axis = 0)), decimals=4)
        var[i] = np.var(dataA, axis=0)


    print("******群平均距離*******")
    print(distance)
    print("******平均群平均距離*******")
    print(sumAll / (num_classes * (num_classes -1)))
    print("******重心間距離*******")
    print(distanceCG)
    print("******平均重心間距離*******")
    print(np.sum(distanceCG) / (num_classes * (num_classes -1)))
    print("******クラスタ毎のデータの分散*******")
    print(np.round(np.average(var, axis=1), decimals=4))


    
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


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/experiments/test/5032AB.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        "-c",
        "--combination",
        default="data/experiments/combination/5032AB.json",
        help="combination(best label to date) file path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()