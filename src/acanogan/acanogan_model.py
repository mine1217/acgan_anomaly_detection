"""
AC-GANにより学習済みのモデルを用いて，入力データの異常度を求めるAC-AnoGANモデル．
"""
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K


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
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='adam')
    intermidiate_model.trainable = False
    # intermidiate_model.summary()
    return intermidiate_model


def sum_of_discrimination(y_true, y_pred, d: Model):
    """
    Discrimination loss.

    Args:
        y_true:正解値
        y_pred:予測値
        d:Discriminator model

    Returns:
        y_true, y_predのDiscrimination loss.
    """
    intermidiate_model = feature_extractor(d)
    y_true = intermidiate_model(y_true)
    y_pred = intermidiate_model(y_pred)
    return K.sum(K.abs(y_true - y_pred))


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
        # Input layer cann't be trained. Add new layer as same size & same
        # distribution
        acanogan_input_data = Input(shape=(input_dim,))
        g_input_data = Dense(
            input_dim,
            activation='tanh',
            trainable=True)(acanogan_input_data)
        label = Input(shape=[1, ], dtype="int32")
        g_out = self.generator([g_input_data, label])
        self.model = Model(inputs=[acanogan_input_data, label], outputs=g_out) ## Model(input=[generatorに入れるノイズ, クラスラベル])
        self.model_weight = None

    def compile(self, optimizer):
        """
        学習するmodelのcompileå

        Args:
            optimizer:使用するoptimizer

        """
        self.model.compile(loss=self.loss, optimizer=optimizer)
        K.set_learning_phase(0)

    def compute_anomaly_score(
            self,
            x: np.array,
            label: np.array,
            iterations: int = 10) -> tuple:
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
        self.z = np.random.randn(1, self.input_dim) ## 正規分布の乱数を100生成
        loss = self.model.fit([self.z, label], x, batch_size=1, 
                              epochs=iterations, verbose=0) ## 異常値計算 batch_sizeはデータ数　一個づつ検証するので1　epochs=一つの訓練データを何回繰り返して学習させるか　今回は10回
        loss = loss.history['loss'][-1]
        generated_data = self.model.predict([self.z, label])

        print(loss)

        return loss, generated_data

    def loss(self, y_true, y_pred):
        """
        Loss function．

        Args:
            y_true:正解値
            y_pred:予測値

        Returns:
            y_true，y_predのLoss．
        """
        return ((1 - self.w) * sum_of_residual(y_true, y_pred)) + \
            (self.w * sum_of_discrimination(y_true, y_pred, self.discriminator))
