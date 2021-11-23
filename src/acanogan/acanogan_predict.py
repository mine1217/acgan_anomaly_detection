"""
提案手法により予測を行う．
"""
import os

from src.acanogan.acanogan_model import ACAnoGAN
import numpy as np


def predict(
        x: np.array,
        g,
        d,
        optim,
        label: np.array,
        iterations: int = 10,
        w: float = 0.1) -> tuple:
    """
    提案手法によりxの異常度を求める．

    Args:
        x:normalized data
        g:Generator model
        d:Discriminator model
        optim:Optimizer
        label:クラスラベル
        iterations:学習回数
        w:Discrimination lossの重み
    Returns:
        異常度，生成データ
    """
    x = x[np.newaxis, :, :]
    acanogan = ACAnoGAN(g=g, d=d, input_dim=100, w=w)
    acanogan.compile(optim)
    anomaly_score, generated_data = acanogan.compute_anomaly_score(
        x=x, label=label, iterations=iterations)
    generated_data = generated_data.flatten()
    return anomaly_score, generated_data
