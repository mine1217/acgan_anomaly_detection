"""
Dの中間層の出力のヒートマップ作成
"""
import _pathmagic
import os

from src.acgan import acgan
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Model
from keras.layers import *
import argparse


def feature_extractor(d: Model, layer_name="d_conv1") -> Model:
    """
    Discriminatorの中間層を出力するモデルを返す．

    Args:
        d:Discriminator model
        layer_name:中間層の名前

    Returns:
        Model:Discriminatorの中間層を出力するモデル．
    """
    intermidiate_model = Model(
        inputs=d.layers[0].input,
        outputs=d.get_layer(layer_name).output)
    # intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
    intermidiate_model.trainable = False
    # intermidiate_model.summary()
    return intermidiate_model


def make_heatmap(
        df: pd.DataFrame, min=0, max=128, num_classes=3,
        model_path="models/experiments/acgan/5032AB/discriminator.h5",
        save_path="output/experiments/discriminator_verification/heatmap/"):
    """

    Args:
        df:
        save_path:Folder to store the heatmap

    Returns:

    """
    # load model

    # AC-Gan model load
    acgan_obj = acgan.ACGAN(
        num_classes=num_classes,
        minimum=min,
        maximum=max
    )

    # encode data
    input, (_, _) = acgan.normalize(df.values)
    input = input[:, :, None]
    input = input.astype(np.float32)

    discriminator = acgan_obj.discriminator
    discriminator.load_weights(model_path)
    encode_model = feature_extractor(discriminator)

    os.makedirs(save_path, exist_ok=True)
    encode_data = encode_model.predict(input)
    # make heatmap
    for i in range(len(encode_data)):
        # plt.subplots(figsize=(4, 10))
        plt.subplots(figsize=(8, 8))
        plt.rcParams["font.size"] = 25
        sns.heatmap(
            encode_data[i].T,
            square=False,
            cbar=False,
            vmin=-0.1,
            vmax=0.1
        )
        # sns.heatmap(
        #     encode_data[i].T,
        #     square=True,
        #     cbar=True)
        # plt.xticks(color="None")
        # plt.xticks([float(i / len(encode_data)) for i in range(5)],
        #            [0, 6, 12, 18, 24], rotation=0)
        plt.xticks([float(i * float(len(encode_data[0]) / 4))
                    for i in range(5)], [0, 6, 12, 18, 24], rotation=0)
        plt.yticks([0, len(encode_data[i][0])], [64, 1])
        plt.tick_params(length=0)
        plt.xlabel("Hour")
        plt.ylabel("Filter")
        plt.savefig(
            save_path +
            df.index[i] +
            ".png",
            bbox_inches='tight',
            pad_inches=0.01)
        # break


def arg_parse():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--input',
        '-i',
        default="data/experiments/test/5032AB_2020_04.csv",
        type=str,
        help='入力データ')
    parser.add_argument(
        '--save',
        '-s',
        default="output/experiments/heatmap/5032AB_2020_04/",
        type=str,
        help='保存先ディレクトリ')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    df = pd.read_csv(
        args.input,
        index_col=0,
        header=None)
    make_heatmap(
        df=df,
        save_path=args.save)


if __name__ == '__main__':
    main()
