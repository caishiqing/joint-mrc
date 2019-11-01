from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

__all__ = ['AdaptiveEmbedding']


class AdaptiveEmbedding(keras.layers.Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.

    # Arguments
        input_dim: int > 0. Size of the vocabulary.
        output_dim: int > 0. Dimension of the dense embedding after projection if it is not equal to embed_dim.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to the `embeddings` matrix.
        mask_zero: Whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using [recurrent layers](recurrent.md)
            which may take variable length input.
            If this is `True` then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 mask_zero=False,
                 return_embeddings=False,
                 return_projections=False,
                 **kwargs):
        super(AdaptiveEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = output_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != input_dim:
                self.cutoffs.append(input_dim)
        self.div_val = div_val
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.return_embeddings = return_embeddings
        self.return_projections = return_projections

        self.embeddings = None
        self.projections = None

    def build(self, input_shape):
        if self.div_val == 1:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.embed_dim),
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                name='embeddings',
            )
            if self.embed_dim != self.output_dim or self.force_projection:
                self.projections = self.add_weight(
                    shape=(self.embed_dim, self.output_dim),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel',
                )
        else:
            self.embeddings, self.projections = [], []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                self.embeddings.append(self.add_weight(
                    shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings-{}'.format(i),
                ))
                projection_shape = (embed_dim, self.output_dim)
                if embed_dim == self.output_dim and not self.force_projection:
                    projection_shape = ()
                self.projections.append(self.add_weight(
                    shape=projection_shape,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='kernel-{}'.format(i),
                ))
        super(AdaptiveEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            output_mask = None
        else:
            output_mask = K.not_equal(inputs, 0)
        if self.return_embeddings or self.return_projections:
            output_mask = [output_mask]
        if self.return_embeddings:
            if self.div_val == 1:
                output_mask += [None]
            else:
                output_mask += [None] * len(self.embeddings)
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_mask += [None]
            else:
                output_mask += [None] * len(self.projections)
        return output_mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape + (self.output_dim,)
        if self.return_embeddings or self.return_projections:
            output_shape = [output_shape]
        if self.return_embeddings:
            if self.div_val == 1:
                output_shape += [K.int_shape(self.embeddings)]
            else:
                output_shape += [K.int_shape(embed) for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    output_shape += [K.int_shape(self.projections)]
            else:
                output_shape += [K.int_shape(proj) for proj in self.projections]
        return output_shape

    def call(self, inputs, **kwargs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        if self.div_val == 1:
            out = K.gather(self.embeddings, inputs)
            if self.embed_dim != self.output_dim or self.force_projection:
                out = K.dot(out, self.projections)
        else:
            out = K.tile(
                K.expand_dims(K.zeros_like(inputs, dtype=K.floatx()), axis=-1),
                (1,) * K.ndim(inputs) + (self.output_dim,),
            )
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                low, high = self.cutoffs[i], self.cutoffs[i + 1]
                mask = K.cast(low <= inputs, K.floatx()) * K.cast(inputs < high, K.floatx())
                selected = K.gather(self.embeddings[i], (inputs - low) * K.cast(mask, 'int32'))
                if embed_dim != self.output_dim or self.force_projection:
                    projected = K.dot(selected, self.projections[i])
                else:
                    projected = selected
                out += projected * K.expand_dims(mask, axis=-1)
        if self.return_embeddings or self.return_projections:
            out = [out]
        if self.return_embeddings:
            if self.div_val == 1:
                out += [self.embeddings]
            else:
                out += [K.identity(embed) for embed in self.embeddings]
        if self.return_projections:
            if self.div_val == 1:
                if self.projections is not None:
                    out += [self.projections]
            else:
                out += [K.identity(proj) for proj in self.projections]
        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'force_projection': self.force_projection,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'mask_zero': self.mask_zero,
            'return_embeddings': self.return_embeddings,
            'return_projections': self.return_projections,
         }
        base_config = super(AdaptiveEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
