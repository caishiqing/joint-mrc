from .backend import keras
from .backend import backend as K

__all__ = ['Identity']


class Identity(keras.layers.Layer):
    """A layer does nothing."""

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return K.stop_gradient(inputs)
