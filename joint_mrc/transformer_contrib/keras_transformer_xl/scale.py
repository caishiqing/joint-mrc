from .backend import keras

__all__ = ['Scale']


class Scale(keras.layers.Layer):
    """Scale all weights.

    # Arguments
        scale: float.
    """

    def __init__(self, scale, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.supports_masking = True
        self.scale = scale

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def get_config(self):
        config = {
            'scale': self.scale,
        }
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
