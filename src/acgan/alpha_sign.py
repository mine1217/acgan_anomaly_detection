from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.engine.base_layer import Layer
from keras import backend as K


class AlphaSign(Layer):
    """
    AlphaSign Activation Function.
    .. math::

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = AlphaSign()(X_input)
    """

    def __init__(self, **kwargs):
        super(AlphaSign, self).__init__(**kwargs)
        self.alpha = 0.2
        self.__name__ = "AlphaSign"
        self.supports_masking = True

    def call(self, inputs):
        return K.relu(inputs / (K.abs(inputs) + (1 / self.alpha)))
