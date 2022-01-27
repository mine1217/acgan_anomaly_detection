"""
AC-GANによる学習済みmodelを用いて異常検知を行う実験用スクリプト．

Example:
    5032AB example

    Normal data experiments
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv

    Anomaly data experiments
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv --is_anomaly_test

    Save variety of things
    ::
        python3 src/experiments/acanogan_test.py --input data/experiments/test/5032AB.csv\
 --score_save output/experiments/score/acanogan/5032AB_normal.csv\
 --generated_save output/experiments/acanogan/generate/5032AB_normal.csv\
 --model_save output/experiments/acanogan/models/5032AB_normal/\

"""
import _pathmagic
import argparse
import json

import os
import numpy as np
import random
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

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data/experiments/test/5032AB.csv",
        help="input file path(for test)")
    parser.add_argument(
        "-m",
        "--model",
        default="models/experiments/gan/5032AB/",
        help="gan generator model file path")
    parser.add_argument(
        "-c",
        "--combination",
        default="data/experiments/combination/5032AB.json",
        help="combination(best label to date) file path")
    parser.add_argument(
        "-mm",
        "--minmax",
        default="data/experiments/minmax/5032AB.json",
        help="data minmax file path")
    parser.add_argument(
        "-it",
        "--iterations",
        type=int,
        default=100,
        help="parameter")
    parser.add_argument(
        "-w",
        "--w",
        type=float,
        default=0.1,
        help="parameter")
    parser.add_argument(
        "--is_anomaly_test",
        "-iat",
        action='store_true',
        help='Flag to test for anomaly data')
    parser.add_argument(
        "-ss",
        "--score_save",
        default="output/experiments/score/acanogan/5032AB_normal.csv",
        help="File path to save the result of anomaly score")
    parser.add_argument(
        "-gs",
        "--generated_save",
        default=None,
        # default="output/experiments/acanogan/generate/5032AB_normal.csv",
        help="File path to save the generated data(If None, do not save)")
    parser.add_argument(
        "-ms",
        "--model_save",
        # default="output/experiments/acanogan/models/5032AB_normal/",
        default=None,
        help="Dir of model and input save(If None, do not save)")
    args = parser.parse_args()
    return args


def random_class_label(class_labels):
    """

    Args:
        class_labels:

    Returns:
        new_class_labels
    """
    max_label = max(class_labels)
    new_class_labels = []
    for c in class_labels:
        while True:
            label = random.randint(0, max_label)
            if c != label:
                new_class_labels.append(label)
                break
    return new_class_labels

def main():
    args = arg_parse()
    # Prepare data
    input = pd.read_csv(args.input, index_col=0, header=None)
    minmax = json.load(open(args.minmax))
    minimum, maximum = minmax["minimum"], minmax["maximum"]
    x_test = input.values

    # Prepare class label
    combination = json.load(open(args.combination))
    combination = list(combination.values())
    day_to_label = optimize.encode_day_to_label(input).values()
    class_labels = [combination[i] for i in day_to_label]
    num_classes = int(max(combination)) + 1
    if args.is_anomaly_test:
        class_labels = random_class_label(class_labels)
    print(class_labels)

    # AC-Gan model load
    acgan_obj = acgan.ACGAN(
        num_classes=num_classes,
        minimum=minimum,
        maximum=maximum
    )
    generator = acgan_obj.generator
    discriminator = acgan_obj.discriminator
    generator.load_weights(args.model + "/generator.h5")
    discriminator.load_weights(args.model + "/discriminator.h5")

    # Data normalize,shape
    sub = maximum - minimum
    if sub == 0:
        # all 0 data
        sub = 1
    x_test = (x_test - minimum) / sub
    x_test = x_test[:, :, None]
    x_test = x_test.astype(np.float32)

    # AC-AnoGan model
    optim = Adam(lr=0.001, amsgrad=True)
    acanogan = ACAnoGAN(g=generator, d=discriminator, input_dim=100, w=args.w)
    acanogan.compile(optim)
    acanogan.model.summary()

    # 各検証用データ用のモデルの初期値一時保存
    init_model_path = "models/experiments/acanogan/init.h5"
    os.makedirs(os.path.dirname(init_model_path), exist_ok=True)
    acanogan.model.save_weights(init_model_path)

    # model_save make dir
    if args.model_save is not None:
        dir_path = args.model_save
        os.makedirs(dir_path, exist_ok=True)
    # Test acanogan
    generated_data_list = np.empty((len(x_test), 120))
    score_list = np.empty(len(x_test))
    z_list = np.empty((len(x_test), 100))
    dates = input.index.to_list()
    i = 0
    for label, test_data in zip(class_labels, x_test):
        acanogan.model.load_weights(init_model_path)
        # anomaly_score, generated_data = predict(
        #     test_data, generator, discriminator, acanogan_optim, label=np.array(
        #         [label]), iterations=args.iterations, w=args.w,

        # Predict
        x = test_data[np.newaxis, :, :]
        anomaly_score, generated_data = acanogan.compute_anomaly_score(
            x=x, label=np.array([label]), iterations=args.iterations)

        # score_list.append(anomaly_score)
        score_list[i] = anomaly_score

        # generated_data = (generated_data * sub) + minimum
        # generated_data_list.append(generated_data)
        if args.generated_save is not None:
            generated_data = generated_data.flatten()
            generated_data_list[i] = generated_data

        if args.model_save is not None:
            z_list[i] = acanogan.z

        # Model save
        if args.model_save is not None:
            acanogan.model.save_weights(f"{args.model_save}{dates[i]}.h")

        # Log
        i += 1
        print(f"data:{i}/{len(x_test)},score:{anomaly_score}")

    # Denormalize
    generated_data_list = (generated_data_list * sub) + minimum

    # Generated save
    if args.generated_save is not None:
        generated_data_df = pd.DataFrame(
            generated_data_list,
            index=dates)
        os.makedirs(os.path.dirname(args.generated_save), exist_ok=True)
        generated_data_df.to_csv(args.generated_save, index=True, header=False)

    # Input save
    if args.model_save is not None:
        os.makedirs(args.model_save, exist_ok=True)
        # noise save
        z_path = f"{args.model_save}/input_z.csv"
        z_df = pd.DataFrame(z_list, index=dates)
        z_df.to_csv(z_path, index=True, header=False)
        # label save
        label_path = f"{args.model_save}/input_label.csv"
        class_labels_df = pd.DataFrame(class_labels, index=dates)
        class_labels_df.to_csv(label_path, index=True, header=False)

    # Anomaly score save
    score_df = pd.DataFrame(score_list, index=dates)
    os.makedirs(os.path.dirname(args.score_save), exist_ok=True)
    score_df.to_csv(args.score_save, index=True, header=False)


if __name__ == '__main__':
    main()
