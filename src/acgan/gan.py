"""
GAN & (LSTM)GAN
(LSTM)GANを使用したい場合は引数の -l のフラグを真にする

AC-GANやCGANとは違い、クラス毎に違うモデルができる
3つクラスがある場合 3つのモデルができあがる

他の部分はだいたいCGANと一緒
"""


import _pathmagic
import collections as cl
import json
import os
import warnings
from keras.models import Sequential, Model
from keras.layers import *
import argparse
import numpy as np
from keras.optimizers import Adam, SGD
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt
from src.acgan import alpha_sign, mish_keras, augment
import tensorflow as tf
import random


class GAN:
    """
    C-GANのモデル.
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
            minimum: int,
            maximum: int,
            batch_size: int = 1,
            seed: int = 0,
            model_save: str = "models/gan/5032AB/0/",
            loss_save: str = "output/ensemble_loss/5032AB.png",
            is_progress_save: bool = False,
            model_progress_save: str = "models/acgan/progress/5032AB/0/",
            l_gan: bool = "false"
    ):
        
        # os.environ['PYTHONHASHSEED'] = '0'
        # session_conf = tf.ConfigProto(
        #     intra_op_parallelism_threads=1,
        #     inter_op_parallelism_threads=1
        # )
        # np.random.seed(seed)
        # random.seed(seed)
        # tf.set_random_seed(seed)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

        self.minimum, self.maximum = minimum, maximum
        self.model_save = model_save
        self.loss_save = loss_save
        # self.is_progress_save = is_progress_save
        # self.model_progress_save = model_progress_save

        self.width = 120
        self.channel = 1
        self.z_size = 100
        self.Goptimizer = Adam(
            0.0002, 0.9
            )
        self.Doptimizer = Adam(
            0.0002, 0.9
            )
        self.start_filters = 512
        self.hidden_dim = 64
        self.batch_size = batch_size
        self.l_gan = l_gan

        # self.losses = [
        #     'binary_crossentropy',
        #     'sparse_categorical_crossentropy']

        # self.loss_weights = [(1 - w), w]
        self.discriminator = self.build_discriminator(l_gan=self.l_gan)
        # self.discriminator.compile(loss=self.losses,
        #                            optimizer=self.optimizer,
        #                            loss_weights=self.loss_weights,
        #                            metrics=['accuracy'])
        self.discriminator.summary()
        self.generator = self.build_generator()
        self.generator.summary()
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        self.set_trainable(self.generator, trainable=False)
        x = Input(shape=(self.width, self.channel))
        z = Input(shape=(self.z_size,))
        # label = Input(shape=[1,], dtype='int32')
        fake = self.generator(z)

         # discriminator takes x and gererated fake
        d_logit_real = self.discriminator(x)
        d_logit_fake = self.discriminator(fake)

        # get loss function
        d_loss, g_loss = self.gan_loss(d_logit_real, d_logit_fake)

        # build trainable discriminator model
        self.D_model = Model([x, z], [d_logit_real, d_logit_fake])
        self.D_model.add_loss(d_loss)
        self.D_model.compile(optimizer=self.Doptimizer, loss=None)

        # freeze discriminator parameter when training discriminator
        self.set_trainable(self.generator, trainable=True)
        self.set_trainable(self.discriminator, trainable=False)

        # build trainable generator model
        self.G_model = Model(z, d_logit_fake)
        self.G_model.add_loss(g_loss)
        self.G_model.compile(optimizer=self.Goptimizer, loss=None)


        # # For the combined model we will only train the generatorM
        # self.discriminator.trainable = False

        # # The discriminator takes generated image as input and determines validity
        # # and the label of that image
        # valid, target_label = self.discriminator(img)

        # # The combined model(stacked generator and discriminator)
        # # Trains the generator to fool the discriminator
        # self.combined = Model([z, label], [valid, target_label])
        # self.combined.compile(
        #     loss=self.losses,
        #     loss_weights=self.loss_weights,
        #     optimizer=self.optimizer)

    def gan_loss(self, d_logit_real, d_logit_fake):
        """
        define loss function
        """

        d_loss_real = K.mean(K.binary_crossentropy(output=d_logit_real,
                                                   target=K.ones_like(
                                                       d_logit_real),
                                                   from_logits=False))
        d_loss_fake = K.mean(K.binary_crossentropy(output=d_logit_fake,
                                                   target=K.zeros_like(
                                                       d_logit_fake),
                                                   from_logits=False))

        d_loss = 0.5*(d_loss_real + d_loss_fake)

        g_loss = K.mean(K.binary_crossentropy(output=d_logit_fake,
                                              target=K.ones_like(d_logit_fake),
                                              from_logits=False))

        return d_loss, g_loss

    def build_generator(self) -> Model:
        """
        Generatorのモデルを定義．

        Returns:
            keras Model:Generator
        """
        z = Input(shape=[self.z_size, ])

        start_filters = 512
        # 2Upsampling adjust
        in_w = int(self.width / 8)
        x = Dense(
            in_w *
            start_filters,
            # activation="relu",
            name="g_dense1")(z)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = Reshape(
            (in_w, start_filters), input_shape=(
                in_w * start_filters,))(x)
        x = Conv1D(
            filters=int(start_filters/2), 
            kernel_size=5, 
            padding="same",
            #activation="relu",
            name="g_conv1")(x)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(
            filters=int(start_filters/4), 
            kernel_size=5, 
            padding="same",
            #activation="relu",
            name="g_conv1_2")(x)
        x = mish_keras.Mish()(x)
        x = BatchNormalization()(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(
            filters=int(start_filters/8), 
            kernel_size=10,
            padding="same",
            # activation="relu",
            name="g_conv1_3")(x)
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
        return Model(z, [outputs], name="generator")

    def build_discriminator(self, l_gan) -> Model:
        """
        Discriminatorのモデルを定義．

        Returns:
            keras Model:Discriminator
        """
        #input = Input(shape=(self.width, self.channel))
        input = Input(shape=(self.width, self.channel))

        if l_gan: 
            x = Bidirectional(CuDNNLSTM(units=self.hidden_dim, return_sequences=True))(input)
            x = mish_keras.Mish()(x)
            x = Bidirectional(CuDNNLSTM(units=self.hidden_dim, return_sequences=True))(x)
            x = mish_keras.Mish()(x)
            x = Bidirectional(CuDNNLSTM(units=int(self.hidden_dim), name='d_conv2'))(x)
            x = mish_keras.Mish()(x)
            x = Dense(64, name='d_conv2')(x)
            x = mish_keras.Mish()(x)
        else:
            start_filters = 64
            x = Conv1D(
                filters=start_filters,
                kernel_size=15,
                strides=2,
                padding="same",
                # activation="relu",
                name="d_conv1")(input)
            x = mish_keras.Mish()(x)
            x = (Dropout(0.25))(x)
            # x = BatchNormalization()(x)
            x = Conv1D(
                filters=start_filters*2,
                kernel_size=10,
                strides=2,
                padding="same",
                # activation="relu",
                name="d_conv1.1")(x)
            x = BatchNormalization()(x)
            x = mish_keras.Mish()(x)
            x = (Dropout(0.25))(x)
            x = Conv1D(
                filters=start_filters*4,
                kernel_size=5,
                strides=2,
                padding="same",
                # activation="relu",
                name="d_conv1.2")(x)
            x = BatchNormalization()(x)
            x = mish_keras.Mish()(x)
            x = (Dropout(0.25))(x)
            x = Conv1D(
                filters=start_filters*8,
                kernel_size=5,
                strides=1,
                padding="same",
                # activation="relu",
                name="d_conv2")(x)
            x = BatchNormalization()(x)
            x = mish_keras.Mish()(x)
            x = (Dropout(0.25))(x)
            x = Flatten()(x)


        # # real or fake
        validity = (Dense(1, activation="sigmoid", name="validity"))(x)
        
        return Model(input, validity, name="discriminator")

    def set_trainable(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

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
            iterations: int = 1000,
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
        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))

        plt_loss = [[],[]]
        print(batch_size)


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
            
            # z = np.random.normal(0, 1, (self.batch_size, self.z_size))
            # z = np.random.normal(0, 1, (batch_size, self.z_size))

            d_loss = self.D_model.train_on_batch([real_data, z], None)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator

            g_loss = self.G_model.train_on_batch(z, None)

            # 折れ線用loss
            plt_loss[0].append(d_loss) 
            plt_loss[1].append(g_loss) 

            # If at save interval => save generated samples,model
            if iteration % interval == 0:
                print(
                    "%d [D loss: %f] [G loss: %f]" %
                    (iteration, d_loss, g_loss))
                # if (g_loss < 0.6) and (d_loss < 0.6):
                #     print("train break")
                #     iterations = iteration
                #     break

                # if self.is_progress_save:
                #     self.save_model_progress(iteration)
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
        #plt.ylim([0.6,0.8])

        plt.savefig(self.loss_save)

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
    parser.add_argument(
        "--l_gan",
        "-lg",
        action='store_true',
        help='Flag to (LSTM)GAN')
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
    num_classes=int(y_train.max()) + 1
    y_train = y_train.reshape((y_train.size,))
    for label in range(num_classes):
        train = x_train[y_train == label]
        gan = GAN(
            minimum=minimum,
            maximum=maximum,
            batch_size = 32,
            model_save=(args.model_save + str(label) + "/"),
            loss_save=(args.loss_save + "_" + str(label) + ".png"),
            is_progress_save=args.is_progress_save,
            model_progress_save=args.model_progress_save,
            l_gan=args.l_gan
        )
        gan.train(x_train=train, iterations=3000, batch_size=32, interval=100)

if __name__ == "__main__":
    main()