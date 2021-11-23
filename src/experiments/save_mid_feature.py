"""
Dの中間層の出力保存
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


def save_mid_feature(
        df: pd.DataFrame, min=0, max=128, num_classes=3,
        model_path="models/experiments/acgan/5032AB/discriminator.h5",
        save_path="output/experiments/discriminator_verification/mid_feature/"):
    """

    Args:
        df:
        model_path:
        save_path:
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

    for i in range(len(encode_data)):
        date = df.index[i]
        data = encode_data[i].T
        np.save(f"{save_path}{date}", data)


def main():
    df = pd.read_csv(
        "data/experiments/test/5032AB_test_normal.csv",
        index_col=0,
        header=None)
    save_mid_feature(
        df=df,
        save_path="output/experiments/discriminator_verification/mid_feature/5032AB/")


if __name__ == '__main__':
    main()
