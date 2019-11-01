from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

__all__ = ['RelativeBias']


class RelativeBias(keras.layers.Layer):
    """Relative bias weights.

    # Arguments
        units: int >= 0. Number of hidden units.

    # Input shape
        Any tensor.

    # Output shape
        Two 1D tensors with shape: `(units,)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self,
                 units,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativeBias, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.bias_context, self.bias_relative = None, None

    def compute_output_shape(self, input_shape):
        return [(self.units,)] * 2

    def compute_mask(self, inputs, mask=None):
        return mask

    def build(self, input_shape):
        self.bias_context = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_context',
        )
        self.bias_relative = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=K.floatx(),
            name='bias_relative',
        )
        super(RelativeBias, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return [
            K.identity(self.bias_context),
            K.identity(self.bias_relative),
        ]

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(RelativeBias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
