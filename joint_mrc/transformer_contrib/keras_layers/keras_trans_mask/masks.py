from .backend import keras
from .backend import backend as K
from .backend import utils

__all__ = ['CreateMask', 'RemoveMask', 'RestoreMask']


class CreateMask(keras.layers.Layer):
    """Create mask from input tensor.
    The shape of the mask equals to the shape of the input tensor.

    # Input shape
        Tensor with shape: `(batch_size, ...)`.

    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, mask_value=0., **kwargs):
        super(CreateMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, self.mask_value)

    def call(self, inputs, **kwargs):
        return K.zeros_like(inputs)

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(CreateMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RemoveMask(keras.layers.Layer):
    """Remove mask from input tensor.

    # Input shape
        Tensor with shape: `(batch_size, ...)`.

    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        return K.identity(inputs)


class RestoreMask(keras.layers.Layer):
    """Restore mask from the second tensor.

    # Input shape
        Tensor with shape: `(batch_size, ...)`.
        Tensor with mask and shape: `(batch_size, ...)`.

    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, **kwargs):
        super(RestoreMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def call(self, inputs, **kwargs):
        return K.identity(inputs[0])

utils.get_custom_objects().update(
    {
        'CreateMask': CreateMask,
        'RemoveMask': RemoveMask,
        'RestoreMask': RestoreMask,
        }
    )
