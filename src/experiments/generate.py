import _pathmagic
import random
import argparse
import json
import os
import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
from keras.optimizers import Adam
from src.acanogan import acanogan_predict
from src.acanogan.acanogan_model import ACAnoGAN
from src.acgan import acgan
from src.preprocess import optimize

import datetime
import jpholiday
from matplotlib import pyplot as plt



def main():
    args = arg_parse()

    #予測に必要なデータロード
    minmax = json.load(open(args.minmax))
    minimum, maximum = minmax["minimum"], minmax["maximum"]

    combination = json.load(open(args.combination))
    combination = list(combination.values())
    num_classes = int(max(combination)) + 1

    iterations = 10

    #データセットを読み込む
    generated_data_list = np.empty((iterations*num_classes, 120))
    
    #generatorロード
    acgan_obj = acgan.ACGAN(
        num_classes=num_classes,
        minimum=minimum,
        maximum=maximum
    )
    
    generator = acgan_obj.generator
    generator.load_weights(args.g_model)
    labels = []
    sub = maximum - minimum
    if sub == 0:
        # all 0 data
        sub = 1

    #予測
    for i in range(num_classes):
        for n in range(iterations):
            noise = np.random.randn(1, 100)
            generated_data_list[i*iterations + n] = generator.predict([noise, np.array([i])]).flatten()
            labels.append("label:" + str(i) + " num:" + str(n))
    
    generated_data_list = (generated_data_list * sub) + minimum




    generated_data_df = pd.DataFrame(
        generated_data_list,
        index=labels)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    generated_data_df.to_csv(args.save, index=True, header=False)

   


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "-s",
        "--save",
        default=None,
        # default="output/experiments/acanogan/generate/5032AB_normal.csv",
        help="File path to save the generated data(If None, do not save)")
    parser.add_argument(
        "-gm",
        "--g_model",
        default="models/experiments/acgan/5032AB/generator.h5",
        help="acgan generator model file path")
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()