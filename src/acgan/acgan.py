"""
ACGANによりinputを学習してモデルを保存する．
num_classes=1を設定した場合，通常のGANとなる．

Example:
    5032AB experiments

    progress_save
    ::
     python3 src/acgan/acgan.py --input data/experiments/train/5032AB.csv --label data/experiments/label/5032AB_train.csv\
 --min_max_save data/experiments/minmax/5032AB.json --model_save models/experiments/acgan/5032AB/\
 --is_progress_save --model_progress_save models/experiments/acgan/progress/5032AB/

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


class ACGAN:
    """
    AC-GANのモデル.
    num_classes=1を設定した場合，通常のGANとなる．

    Attributes:
        num_classes:クラス数
        minimum:min
        maximum:max
        w:class loss weight
        model_save:学習後のモデルの保存先
        is_progress_save:学習の途中経過を保存するかどうか．
        model_progress_save:学習途中のモデルの保存先．
    """

    def __init__(
            self,
            num_classes: int,
            minimum: int,
            maximum: int,
            w: float = 0.1,
            model_save: str = "models/acgan/5032AB/",
            is_progress_save: bool = False,
            model_progress_save: str = "models/acgan/progress/5032AB/",
    ):
        self.num_classes = num_classes
        self.minimum, self.maximum = minimum, maximum
        if self.num_classes == 1:
            self.w = 0
        else:
            self.w = w
        self.model_save = model_save
        self.is_progress_save = is_progress_save
        self.model_progress_save = model_progress_save
        self.width = 120
        self.channel = 1
        self.z_size = 100
        self.optimizer = Adam(0.0002, 0.5)
        self.losses = [
            'binary_crossentropy',
            'sparse_categorical_crossentropy']

        self.loss_weights = [(1 - w), w]
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.losses,
                                   optimizer=self.optimizer,
                                   loss_weights=self.loss_weights,
                                   metrics=['accuracy'])
        self.discriminator.summary()
        self.generator = self.build_generator()
        self.generator.summary()
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        z = Input(shape=(self.z_size,))
        label = Input(shape=(1,))
        img = self.generator([z, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model(stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, label], [valid, target_label])
        self.combined.compile(
            loss=self.losses,
            loss_weights=self.loss_weights,
            optimizer=self.optimizer)

    def build_generator(self) -> Model:
        """
        Generatorのモデルを定義．

        Returns:
            keras Model:Generator
        """
        z = Input(shape=[self.z_size, ])
        label = Input(shape=[1, ], dtype="int32")

        label_embedding = Flatten()(Embedding(
            self.num_classes,
            self.z_size)(label))
        model_input = multiply([z, label_embedding])
        start_filters = 256
        # 2Upsampling adjust
        in_w = int(self.width / 4)
        x = Dense(
            in_w *
            start_filters,
            # activation="tanh",
            name="g_dense1")(model_input)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = Reshape(
            (in_w, start_filters), input_shape=(
                in_w * start_filters,))(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=64, kernel_size=5, padding="same",
                   # activation="tanh",
                   name="g_conv1")(x)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(
            filters=1,
            kernel_size=15,
            padding="same",
            # original_activations="tanh",
            name="g_conv2")(x)
        outputs = Activation(alpha_sign.AlphaSign())(x)
        # model.add(BatchNormalization())
        # model.add(Flatten())
        # model.add(Dense(self.width, original_activations="relu", name="out"))
        # model.add(Reshape((self.width, 1), input_shape=(self.width,)))
        return Model([z, label], [outputs], name="generator")

    def build_discriminator(self) -> Model:
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

        # real or face
        validity = Dense(1, activation="sigmoid", name="validity")(features)
        # auxiliary classifier
        classifier = Dense(
            self.num_classes,
            activation="softmax",
            name="classifier")(features)
        return Model([input], [validity, classifier], name="discriminator")

    def save_model_progress(self, iteration: int):
        """
        学習途中のmodelを保存する．

        Args:
            iteration:学習回数．
        """
        dir_path = self.model_progress_save
        os.makedirs(dir_path, exist_ok=True)
        # self.combined.save_weights(dir_path + str(iteration) + 'iteration.h5')
        self.generator.save_weights(
            dir_path + str(iteration) + 'iteration_g.h5')
        self.discriminator.save_weights(
            dir_path + str(iteration) + 'iteration_d.h5')

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
            interval:is_progress_save=Trueの時に，保存を行うinterval．
        """
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        plt_loss = [[],[]]

        for iteration in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_data = x_train[idx]

            # Real data augmentation
            real_data = augment.augmentation(real_data)

            # Sample noise as generator input
            # z = np.random.uniform(-1, 1, size=(batch_size, self.z_size))
            z = np.random.randn(batch_size, self.z_size)
            # z = np.random.normal(0, 1, (batch_size, self.z_size))

            # The labels of the digits that the generator tries to create an
            # data representation of
            if self.num_classes == 1:
                # all same label
                fake_labels = np.zeros((batch_size, 1), dtype=int)
            else:
                fake_labels = np.random.randint(
                    0, self.num_classes - 1, (batch_size, 1))

            # Generate a half batch of new data
            gen_data = self.generator.predict([z, fake_labels])
            
            # Real data labels.
            real_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                real_data, [valid, real_labels])
            d_loss_fake = self.discriminator.train_on_batch(
                gen_data, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator
            g_loss = self.combined.train_on_batch(
                [z, fake_labels], [valid, fake_labels])

            # 折れ線用loss
            plt_loss[0].append(d_loss[0]) 
            plt_loss[1].append(g_loss[0]) 

            # If at save interval => save generated samples,model
            if iteration % interval == 0:
                print(
                    "%d [D loss: %f] [G loss: %f]" %
                    (iteration, d_loss[0], g_loss[0]))
                if self.is_progress_save:
                    self.save_model_progress(iteration)
        # ---------------------
        #  Save the model after training
        # ---------------------
        dir_path = self.model_save
        os.makedirs(dir_path, exist_ok=True)
        # self.combined.save_weights(dir_path + self.data_id + '.h5')
        self.generator.save_weights(dir_path + 'generator.h5')
        self.discriminator.save_weights(dir_path + 'discriminator.h5')

        plt.figure(dpi=500)

        plt.plot(range(0, iterations), plt_loss[0][:iterations], linewidth=1, label="d_loss", color="red")
        plt.plot(range(0, iterations), plt_loss[1][:iterations], linewidth=1, label="g_loss", color="blue")
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
        "--is_progress_save",
        "-ips",
        action='store_true',
        help='Flag to save the progress of the model or sample plot．default=False')
    parser.add_argument(
        "--model_progress_save",
        "-mps",
        default="models/acgan/progress/5032AB/",
        type=str,
        help="Dir path to save the progress of the model．")
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
    acgan = ACGAN(
        num_classes=int(
            y_train.max()) + 1,
        minimum=minimum,
        maximum=maximum,
        w=0.1,
        model_save=args.model_save,
        is_progress_save=args.is_progress_save,
        model_progress_save=args.model_progress_save)
    acgan.train(x_train, y_train, iterations=2000, batch_size=32,
                interval=100)


if __name__ == "__main__":
    main()
