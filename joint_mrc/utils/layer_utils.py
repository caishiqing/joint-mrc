from . import keras, utils, activations, initializers
from . import backend as K
import tensorflow as tf

__all__ = [
    'Pointer', 'MaskQuest', 'focal_loss',
]

class Pointer(keras.layers.Layer):
    """Pointer Network Layer
    Implement a linear classifier for elements of a sequence considering masking.
    Args:
        mode: Output mode, if 'categorical' then the question is which one
              in sequence steps, else if 'binary' denotes harf point.
    """
    def __init__(self, mode='categorical', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 **kwargs):
        super(Pointer, self).__init__(**kwargs)
        if mode == 'binary':
            self.activation = activations.get('sigmoid')
        elif mode == 'categorical':
            self.activation = activations.get('linear')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias
        self.mode = mode
        self.supports_masking = True

    def build(self, input_shape):
        _, length, dim = input_shape
        if not isinstance(dim, int):
            dim = dim.value
        self.kernel = self.add_weight(
                                name='kernel',
                                shape=[dim, 1],
                                initializer=self.kernel_initializer,
                                )
        if self.use_bias:
            self.bias = self.add_weight(
                                name='bias',
                                shape=[1],
                                initializer=self.bias_initializer,
                                )
        else:
            self.bias = None
        super(Pointer, self).build(input_shape)

    def call(self, inputs, mask=None):
        y = K.dot(inputs, self.kernel)
        if self.use_bias:
            y = K.bias_add(y, self.bias)
        y = self.activation(y)
        y = K.squeeze(y, -1)
        mask = K.cast(mask, y.dtype)
        if self.mode == 'binary':
            p = y * mask
        elif self.mode == 'categorical':
            logits = K.exp(y)
            logits -= K.max(logits, axis=-1, keepdims=True)
            logits *= mask
            p = logits / K.sum(logits, axis=-1, keepdims=True)
            
        return p

    def get_cinfig(self):
        config = super(Pointer, self).get_config()
        config['mode'] = self.mode
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def compute_mask(self, inputs, mask=None):
        return None


class MaskQuest(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskQuest, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        seq, seg = inputs
        return seq

    def compute_mask(self, inputs, mask=None):
        seq, seg = inputs
        if mask is None:
            return K.cast(seg, 'bool')
        mask = mask[0]
        mask = mask & K.cast(seg, 'bool')
        return mask


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return - K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
                    K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

utils.get_custom_objects().update(
    {
        'Pointer': Pointer,
        'MaskQuest': MaskQuest,
        'focal_loss': focal_loss,
        }
    )