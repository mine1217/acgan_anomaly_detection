"""
実験に使用するデータをumapを使って散布図にする.
あとクラスタ間距離を求めてターミナルに出力する.

Example:
    5032AB example
    ::
    python3 -i src/preprocess/umap_ex.sh -i "data/processed/experiments/5032AB.csv" -c data/experiments/combination/5032AB.json" -s "output/experiments/umap/5032AB.png"
"""

import datetime
import jpholiday
import json
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import umap.umap_ as umap
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm

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

    input = pd.read_csv(args.input, header=None, index_col=0)

    combination = json.load(open(args.combination))
    combination = list(combination.values())
    class_labels = encode_day_to_label(input).values()

    if args.weekday_label:
        class_labels = [combination[i] for i in class_labels]
    else:
        class_labels = list(class_labels)

    #Encode label
    label = encode_day_to_label(input)

    #データセットを読み込む
    dataset = datasets.load_digits()
    X, y = input.values, class_labels

    #もし正規化するなら
    if args.normalize:
        X, (minimum, maximum) = normalize(X)

    print(X)

    A=0
    B=0
    C=0
    for a in class_labels:
        if (a == 0):
            A+=1
        elif (a == 1):
            B+=1
        elif(a == 2):
            C+=1
    print(A,B,C)


    #次元削減する
    mapper = umap.UMAP(random_state=0)
    embedding = mapper.fit_transform(X)

    plt.figure(dpi=500)

    #結果を二次元でプロットする
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    if args.day_label:
        usercmap = plt.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=0, vmax=embedding_x.size)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=usercmap)
        for i, x in enumerate(embedding_x):
            c = scalarMap.to_rgba(i)
            plt.scatter(x,embedding_y[i],color=c)
    else:
        for n in np.unique(y):
            plt.scatter(embedding_x[y == n],
                        embedding_y[y == n],
                        label=n)

    num_classes = int(max(combination)) + 1
    x = embedding
    distance = np.zeros((num_classes , num_classes))
    distanceCG = np.zeros((num_classes , num_classes))
    var = np.zeros((num_classes, 2))
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
    
    # if args.day_label:
    #     day_labels = input.index.to_list()

    #     for i, d in enumerate(day_labels):
    #         plt.annotate(d, (embedding_x[i],embedding_y[i]), fontsize=4)
        

    #グラフを表示する
    plt.grid()
    if not (args.day_label):
        plt.legend()
    #plt.show()
    plt.savefig(args.save)
    print(args.save)

def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/processed/train/5032AB.csv",
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
        default="output/experiments/umap/5032AB.png",
        help="File to save the roc curve")
    parser.add_argument(
        "-d",
        "--day_label",
        action='store_true',
        help="Plot day label")
    parser.add_argument(
        "-n",
        "--normalize",
        action='store_true',
        help="Plot day label")
    parser.add_argument(
        "-w",
        "--weekday_label",
        action='store_false',
        help="class or day of week")
    
    args = parser.parse_args()
    return args

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


if __name__ == '__main__':
    main()