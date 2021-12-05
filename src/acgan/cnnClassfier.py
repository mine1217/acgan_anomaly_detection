"""
ACGANによりinputを学習してモデルを保存する．
num_classes=1を設定した場合，通常のGANとなる．

Example:
    5032AB experiments

    progress_save
    ::
     python3 src/acgan/acgan.py --input data/experiments/train/5032AB.csv --label data/experiments/label/5032AB_train.csv\
 --min_max_save data/experiments/minmax/5032AB.json --model_save models/experiments/acgan/5032AB/\

    Not progress_save
    ::
     python3 src/acgan/acgan.py --input data/experiments/train/5032AB.csv --label data/experiments/label/5032AB_train.csv\
 --min_max_save data/experiments/minmax/5032AB.json --model_save models/experiments/acgan/5032AB/
 """

import _pathmagic
import collections as cl
import json
import os
import warnings
from keras.models import Model
from keras.layers import *
import argparse
import numpy as np
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from src.acgan import alpha_sign, mish_keras, augment


class cnnClassfier:
    """
    AC-GANのモデル.
    num_classes=1を設定した場合，通常のGANとなる．

    Attributes:
        num_classes:クラス数
        minimum:min
        maximum:max
        w:class loss weight
        model_save:学習後のモデルの保存先
    """

    def __init__(
            self,
            num_classes: int,
            minimum: int,
            maximum: int,
            w: float = 0.5,
            model_save: str = "models/cnnClassfier/5032AB/",
    ):
        self.num_classes = num_classes
        self.minimum, self.maximum = minimum, maximum
        if self.num_classes == 1:
            self.w = 0
        else:
            self.w = w
        self.model_save = model_save

        self.width = 120
        self.channel = 1
        self.z_size = 100
        self.optimizer = Adam()
        self.losses = 'sparse_categorical_crossentropy'

        self.classfier = self.build_classfier()
        self.classfier.compile(loss=self.losses,
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.classfier.summary()

    def build_classfier(self) -> Model:
        """
        Discriminatorのモデルを定義．

        Returns:
            keras Model:Discriminator
        """
        input = Input(shape=(self.width, self.channel))
        x = Conv1D(
            filters=64,
            kernel_size=15,
            strides=2,
            padding="same",
            # activation="relu",
            name="d_conv1")(input)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = Conv1D(
            filters=128,
            kernel_size=5,
            strides=2,
            padding="same",
            # activation="relu",
            name="d_conv2")(x)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(units=1024,
                  # activation="relu",
                  name="d_dense1")(x)
        x = mish_keras.Mish()(x)
        features = BatchNormalization()(x)

        # auxiliary classifier
        classifier = Dense(
            self.num_classes,
            activation="softmax",
            name="classifier")(features)
        return Model([input], classifier, name="classfier")

    def train(
            self,
            x_train: np.array,
            y_train: np.array,
            iterations: int = 3000,
            batch_size: int = 32,
            interval: int = 100):
        """

        Args:
            x_train:学習データ
            y_train:学習用データセットをクラスラベル
            iterations:学習回数
            batch_size:バッチサイズ
        """
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        plt_loss = []

        for iteration in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_data = x_train[idx]

            # Real data augmentation
            real_data = augment.augmentation(real_data)
            
            # Real data labels.
            real_labels = y_train[idx]

            # Train the classfier
            loss = self.classfier.train_on_batch(real_data, real_labels)
            print(loss)


            # 折れ線用loss
            plt_loss.append(loss[0]) 

            # If at save interval => save generated samples,model
            if iteration % interval == 0:
                print(
                    "%d [C loss: %f]" %
                    (iteration, loss[0]))

        # ---------------------
        #  Save the model after training
        # ---------------------
        dir_path = self.model_save
        os.makedirs(dir_path, exist_ok=True)
        # self.combined.save_weights(dir_path + self.data_id + '.h5')
        self.classfier.save_weights(dir_path + '.h5')

        plt.figure(dpi=500)

        plt.plot(range(0, len(plt_loss)), plt_loss, linewidth=1, label="d_loss", color="red")
        plt.xlabel('iteration')
        plt.ylabel('loss') 
        plt.legend()

        args = arg_parse()

        plt.savefig(args.loss_save)

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


def denormalize(x: np.array, minimum: int, maximum: int) -> np.array:
    """
    Denormalize．

    Args:
        x:Input data
        minimum:Min
        maximum:Max

    Returns:
        Denormalize data
    """
    return x * (maximum - minimum) + minimum


def preprocess_train_data(input_path: str, label_path: str) -> tuple:
    """
    input_path，label_pathのcsvをmodelに入力できる形式に変換する．また，min maxを求めて返す．
    Args:
        input_path:Input csv path
        label_path:Label csv path
    Returns:
        x_train, minimum, maximum, y_train
    """

    input = pd.read_csv(input_path, index_col=0, header=None)
    label = pd.read_csv(label_path, index_col=0, header=None)
    # [0, 1] normalize
    x_train, (minimum, maximum) = normalize(input.values)
    # dimensional expansion for channel
    x_train = x_train[:, :, None]
    x_train = x_train.astype(np.float32)
    y_train = label.values.reshape(-1, 1)
    return x_train, minimum, maximum, y_train


def save_minmax(minimum: int, maximum: int, save_path: str):
    """
    Min max 保存．

    Args:
        minimum:Min
        maximum:Max
        save_path:保存先path
    """
    minmax_data = cl.OrderedDict()
    minmax_data['minimum'] = minimum
    minmax_data['maximum'] = maximum
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fw = open(save_path, 'w')
    json.dump(minmax_data, fw, indent=4)

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/train/5032AB.csv",
        type=str,
        help="input path")
    parser.add_argument(
        "--label",
        "-l",
        default="data/processed/label/5032AB.csv",
        type=str,
        help="Label path．")
    parser.add_argument(
        "--min_max_save",
        "-mms",
        default="data/processed/minmax/5032AB.json",
        type=str,
        help="Json path to save min max．")
    parser.add_argument(
        "--model_save",
        "-ms",
        default="models/acgan/5032AB/",
        type=str,
        help="Dir path to save model．")
    parser.add_argument(
        "--loss_save",
        "-ls",
        default="output/experiments/acgan_loss/5032AB.png",
        type=str,
        help="Dir path to save acgan_loss．")
    args = parser.parse_args()
    return args


def main():
    warnings.simplefilter('ignore')
    args = arg_parse()
    x_train, minimum, maximum, y_train = preprocess_train_data(
        args.input, args.label)
    save_minmax(
        minimum,
        maximum,
        save_path=args.min_max_save)
    cnn = cnnClassfier(
        num_classes=int(
            y_train.max()) + 1,
        minimum=minimum,
        maximum=maximum,
        w=0.1,
        model_save=args.model_save)
    cnn.train(x_train, y_train, iterations=2000, batch_size=32,
                interval=100)


if __name__ == "__main__":
    main()
