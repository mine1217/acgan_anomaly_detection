import _pathmagic
import random
import argparse
import json
import os

import numpy as np
import pandas as pd
from keras.optimizers import Adam
from src.acanogan import acanogan_predict
from src.acanogan.acanogan_model import ACAnoGAN
from src.acgan import cnnClassfier
from src.preprocess import optimize

import datetime
import jpholiday
from matplotlib import pyplot as plt
import umap.umap_ as umap



def main():
    args = arg_parse()

    #予測に必要なデータロード
    
    input = pd.read_csv(args.input, header=None, index_col=0)
    minmax = json.load(open(args.minmax))
    minimum, maximum = minmax["minimum"], minmax["maximum"]

    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = encode_day_to_label(input).values()
    class_labels = [combination[i] for i in day_to_label]
    print(class_labels)
    num_classes = int(max(combination)) + 1

    #Encode label
    label = encode_day_to_label(input)

    #データセットを読み込む
    X, y = input.values, class_labels
    dates = input.index.to_list()
    input_data = np.reshape(X, [len(X),120,1])
    
    #discriminatorロード
    cnn = cnnClassfier.cnnClassfier(
        num_classes=num_classes,
        minimum=minimum,
        maximum=maximum
    )
    classfier = cnn.classfier
    classfier.load_weights(args.model)

    #予測
    output = classfier.predict(input_data)
    p_tf, p_class_prob = output[0], np.array(output[1])
    y = []
   
    #所属確立の高いindexをクラスにする
    for cp in p_class_prob:
        y.append(np.argmax(cp))
    
    print(y)
    
    #UMAP
    plt.figure(dpi=500)

    #次元削減する
    mapper = umap.UMAP(random_state=0)
    embedding = mapper.fit_transform(X)

    #結果を二次元でプロットする
    embedding_x = embedding[:, 0]
    embedding_y = embedding[:, 1]
    for n in np.unique(y):
        plt.scatter(embedding_x[y == n],
                    embedding_y[y == n],
                    label=n)
        

    #グラフを表示する
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig(args.save)

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
        "-m",
        "--model",
        default="models/experiments/cnn/5032AB/classfier.h5",
        help="cnn model file path")
    parser.add_argument(
        "-mm",
        "--minmax",
        default="data/experiments/minmax/5032AB.json",
        help="data minmax file path")
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