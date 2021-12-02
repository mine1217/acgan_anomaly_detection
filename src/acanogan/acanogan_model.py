"""
AC-GANにより学習済みのモデルを用いて，入力データの異常度を求めるAC-AnoGANモデル．
"""
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




from keras.models import Model
from keras.layers import *
import keras.backend as K
from src.acgan import mish_keras

def feature_extractor(d: Model, layer_name="d_conv2") -> Model:
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
    intermidiate_model.compile(
        # loss='sparse_categorical_crossentropy', 
        loss='binary_crossentropy', 
        optimizer='adam')
    # intermidiate_model.summary()
    return intermidiate_model


# def sum_of_discrimination(y_true, y_pred, d: Model):
#     """
#     Discrimination loss.

#     Args:
#         y_true:正解値
#         y_pred:予測値
#         d:Discriminator model

#     Returns:
#         y_true, y_predのDiscrimination loss.
#     """
#     intermidiate_model = feature_extractor(d)
#     y_true = intermidiate_model(y_true)
#     y_pred = intermidiate_model(y_pred)
#     return K.sum(K.abs(y_true - y_pred))


def sum_of_residual(y_true, y_pred):
    """
        Residual loss.

        Args:
            y_true:正解値
            y_pred:予測値

        Returns:
            y_true, y_predのResidual loss.
    """
    return K.sum(K.abs(y_true - y_pred))


class ACAnoGAN:
    """
    AC-GANにより学習済みのモデルを用いて，入力データの異常度を求めるモデルAC-AnoGANの定義．

    Attributes:
        input_dim:ランダムノイズの次元数
        g:Generator model
        d:Discriminator model
        w:Discrimination lossの重み
    """

    def __init__(
            self,
            g: Model,
            d: Model,
            input_dim: int = 100,
            w: float = 0.1):
        self.input_dim = input_dim
        self.generator = g
        self.discriminator = d
        self.w = w
        # self.z = np.random.randn(1, self.input_dim)
        self.z = None
        self.generator.trainable = False
        # Input layer cann't be trained. Add new layer as same size & same
        # distribution
        acanogan_input_data = Input(shape=(input_dim,)) 
        g_input_data = Dense(
            input_dim,
            # activation='tanh',
            trainable=True)(acanogan_input_data)
        #g_input_data = mish_keras.Mish()(g_input_data)
        # g_input_data = LeakyReLU(0.5)(g_input_data)
        g_input_data = BatchNormalization()(g_input_data)
        label = Input(shape=[1, ], dtype="int32")

        intermidiate_model = feature_extractor(self.discriminator)
        intermidiate_model.trainable = False

        g_out = self.generator([g_input_data, label])
        d_out = intermidiate_model(g_out)

        self.model = Model(inputs=[acanogan_input_data, label], outputs=[g_out, d_out])


    def compile(self, optimizer):
        """
        学習するmodelのcompileå

        Args:
            optimizer:使用するoptimizer

        """
        self.model.compile(loss=sum_of_residual, loss_weights= [1-self.w, self.w], optimizer=optimizer)
        K.set_learning_phase(0)

    def compute_anomaly_score(
            self,
            x: np.array,
            label: np.array,
            iterations: int = 100) -> tuple:
        """
        xの異常度を求める．

        Args:
            x:異常判定を行うデータ．
            label:xのクラスラベル
            iterations:学習回数

        Returns:
            異常度，生成データ
        """
        # z = np.random.uniform(-1, 1, size=(1, self.input_dim))
        # z = Input((100,), name="Z")
        # learning for changing latent
        self.z = np.random.randn(1, self.input_dim)

        intermidiate_model = feature_extractor(self.discriminator)
        d_x = intermidiate_model.predict(x)

        loss = self.model.fit([self.z, label], [x, d_x], batch_size=1,
                              epochs=iterations, verbose=0)
        loss = loss.history['loss'][-1]
        generated_data, _ = self.model.predict([self.z, label])

        return loss, generated_data

    # def loss(self, y_true, y_pred):
    #     """
    #     Loss function．

    #     Args:
    #         y_true:正解値
    #         y_pred:予測値

    #     Returns:
    #         y_true，y_predのLoss．
    #     """
    #     return ((1 - self.w) * sum_of_residual(y_true, y_pred)) + \
    #         (self.w * sum_of_discrimination(y_true, y_pred, self.discriminator))
