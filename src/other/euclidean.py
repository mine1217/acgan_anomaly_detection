"""
データ間のユークリッド距離を求める．

Example:
    5032AB example
    ::
        python3 src/other/euclidean.py 
            -iA data/experiments/test/5032AB.csv
            -iB output/experiments/generated/5032AB_normal_acgan_1.csv 
            -c data/experiments/combination/5032AB.json 
            -s output/experiments/euclidean/5032AB_acgan_1.csv
"""
import _pathmagic
import random
import argparse
import json
import os

import datetime
import jpholiday
import json

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import umap.umap_ as umap


def normalize(x: np.array, maximum, minimum) -> tuple:
    """
    Min max normalize．

    Args:
        x:input data
    Returns:
        x_train, minimum, maximum
    """
    return (x - minimum) / (maximum - minimum), (minimum, maximum)

def main():
    args = arg_parse()
    
    A = pd.read_csv(args.inputA, header=None, index_col=0)
    B = pd.read_csv(args.inputB, header=None, index_col=0)
    dates = np.array(A.index)


    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = encode_day_to_label(A).values()
    y = [combination[i] for i in day_to_label]

    minimum = A.values.min(axis=None)
    maximum = A.values.max(axis=None)
    print(maximum)
    A, (minimum, maximum) = normalize(A.values, maximum, minimum)
    B, (minimum, maximum) = normalize(B.values, maximum, minimum)
    print(A)

    dist = np.empty(len(A))
    for i, a in enumerate(A):
        dist[i] = np.linalg.norm((a)-(B[i]))
    calc = dist

    for i in np.unique(y):
        dates = np.append(dates, 'class_' + str(i) + '_avg')
        dist = np.append(dist, np.average(calc[y == i]))
        print(i)

    dates = np.append(dates, 'avg')
    dates = np.append(dates, 'mdn')
    dist = np.append(dist, np.average(calc))
    dist = np.append(dist, np.median(calc))


    dist = pd.DataFrame(
        dist,
        index=dates)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    dist.to_csv(args.save, index=True, header=False)

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
        '--inputA',
        '-iA',
        default="data/experiments/test/5032AB.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        '--inputB',
        '-iB',
        default="output/experiments/generated/5032AB_normal_acgan_1.csv",
        type=str,
        help='input csv path')
    parser.add_argument(
        "-c",
        "--combination",
        default="data/experiments/combination/5032AB.json",
        help="combination(best label to date) file path")
    parser.add_argument(
        "-s",
        "--save",
        default="output/experiments/euclidean/5032AB_acgan_1.csv",
        help="File to save the roc curve")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()