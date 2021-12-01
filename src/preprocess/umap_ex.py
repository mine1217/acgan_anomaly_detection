import datetime
import jpholiday
import json
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
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
    for n in np.unique(y):
        plt.scatter(embedding_x[y == n],
                    embedding_y[y == n],
                    label=n)
    
    if args.day_label:
        day_labels = input.index.to_list()

        for i, d in enumerate(day_labels):
            plt.annotate(d, (embedding_x[i],embedding_y[i]), fontsize=4)
        

    #グラフを表示する
    plt.grid()
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