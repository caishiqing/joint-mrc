import tensorflow as tf
from .backend import keras
from .backend import backend as K

__all__ = ['Memory']


class Memory(keras.layers.Layer):
    """Positional embeddings.

    # Arguments
        batch_size: int > 0. Maximum batch size.
        memory_len: int > 0. Maximum memory length.
        target_len: int > 0. Maximum length of targets.
        output_dim: int > 0. Dimension of outputs.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        1D tensor with shape: `(batch_size,)` represents length of memory.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length + memory_length, output_dim)`.

    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self, batch_size, memory_len, target_len, output_dim, **kwargs):
        super(Memory, self).__init__(**kwargs)
        self.supports_masking = True
        self.stateful = True

        self.batch_size = batch_size
        self.memory_len = memory_len
        self.target_len = target_len
        self.output_dim = output_dim

        self.memory = None

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_len + self.target_len, self.output_dim),
            initializer='zeros',
            trainable=False,
            name='memory',
        )
        super(Memory, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.batch_size, None, self.output_dim

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, **kwargs):
        inputs, memory_length = inputs
        memory_length = K.cast(memory_length[0][0], 'int32')
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')

        # Build new memory
        pad = K.tile(inputs[0:1, ...], (self.batch_size - batch_size, 1, 1))
        padded = K.concatenate([inputs, pad], axis=0)              # (self.batch_size, seq_len, output_dim)
        new_memory = K.concatenate([self.memory, padded], axis=1)  # (self.batch_size, self.memory_len + seq_len, ...)
        new_memory = tf.slice(                                     # (self.batch_size, self.memory_len, output_dim)
            new_memory,
            (0, seq_len, 0),
            (self.batch_size, self.memory_len + self.target_len, self.output_dim),
        )
        self.add_update(K.update(self.memory, new_memory), inputs)

        # Build output
        old_memory = tf.slice(                                     # (batch_size, memory_length, output_dim)
            new_memory,
            (0, K.maximum(0, self.memory_len + self.target_len - seq_len - memory_length), 0),
            (batch_size, K.minimum(self.memory_len, memory_length), self.output_dim),
        )

        return old_memory

    def get_config(self):
        config = {
            'batch_size': self.batch_size,
            'memory_len': self.memory_len,
            'target_len': self.target_len,
            'output_dim': self.output_dim,
        }
        base_config = super(Memory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
