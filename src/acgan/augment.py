"""
Waveform data augmentation
"""
import numpy as np


def augmentation(data: np.array) -> np.array:
    """
    Data augmentation.
    Args:
        data:input data

    Returns:
        np.array:Augmented data
    """
    #np.random.seed(0)
    data = add_random_noise(data)
    data = width_shift_range(data)
    return data


def add_random_noise(data: np.array) -> np.array:
    """
    データにランダムにノイズを付与する．
    ただし，0の要素はそのまま．
    Args:
        data:input data

    Returns:
        np.array:Augmented data
    """
    non_zero = data[data > 0]
    noise = np.random.normal(0, scale=non_zero.std(), size=data.shape)
    data = np.where(data == 0, 0, data + noise * 0.01)
    return data


def width_shift_range(data: np.array) -> np.array:
    """
    左右に平行移動．
    Args:
        data:input data

    Returns:
        np.array:Augmented data
    """
    data = np.insert(data, -1, 0, axis=1)
    data = np.insert(data, 0, 0, axis=1)

    shift_range = np.random.randint(-2, 2)
    data = np.roll(data, shift_range, axis=1)
    data = np.delete(data, [0, len(data)], axis=1)
    return data
