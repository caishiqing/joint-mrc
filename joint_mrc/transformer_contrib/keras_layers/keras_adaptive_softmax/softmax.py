from .backend import keras, initializers, regularizers, constraints
from .backend import backend as K

__all__ = ['AdaptiveSoftmax']


class AdaptiveSoftmax(keras.layers.Layer):
    """Turns dense vectors into probabilities.

    # Arguments
        input_dim: int > 0. Dimension of input vectors.
        output_dim: int > 0. Number of output classes.
        embed_dim: int > 0. Dimension of the dense embedding.
        cutoffs: list of ints. Indices of splitting points.
        div_val: int >= 0. The scaling parameter of embedding.
        use_bias: Boolean. Whether to bias terms.
        force_projection: Boolean. Add projection even if output_dim equals to embed_dim.
        bind_embeddings: list of boolean. Whether to use the existed embeddings as mapping.
        bind_projections: list of boolean. Whether to use the existed projections as mapping.

    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309.pdf)
    """

    def __init__(self, input_dim, output_dim, embed_dim=None,
                 cutoffs=None, div_val=1, use_bias=True,
                 force_projection=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 bind_embeddings=False,
                 bind_projections=False,
                 **kwargs):
        super(AdaptiveSoftmax, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if embed_dim is None:
            self.embed_dim = input_dim
        self.cutoffs = cutoffs
        if cutoffs is not None:
            if self.cutoffs[0] != 0:
                self.cutoffs = [0] + self.cutoffs
            if self.cutoffs[-1] != output_dim:
                self.cutoffs.append(output_dim)
        self.div_val = div_val
        self.use_bias = use_bias
        self.force_projection = force_projection
        if force_projection is None:
            if div_val == 1:
                self.force_projection = False
            else:
                self.force_projection = True
        self.cluster_num = 0
        if self.cutoffs is not None:
            self.cluster_num = len(self.cutoffs) - 2

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.bind_embeddings = bind_embeddings
        if not isinstance(bind_embeddings, list):
            self.bind_embeddings = [bind_embeddings] * (self.cluster_num + 1)
        self.bind_projections = bind_projections
        if not isinstance(bind_projections, list):
            self.bind_projections = [bind_projections] * (self.cluster_num + 1)

        self.embeddings, self.projections, self.biases = (None,) * 3
        self.kernel_cluster, self.bias_cluster = None, None

    def build(self, input_shape):
        if self.div_val == 1:
            if not self.bind_embeddings[0]:
                self.embeddings = self.add_weight(
                    shape=(self.output_dim, self.embed_dim),
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint,
                    name='embeddings',
                )
            if self.embed_dim != self.input_dim or self.force_projection:
                if not self.bind_projections[0]:
                    self.projections = self.add_weight(
                        shape=(self.embed_dim, self.input_dim),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        name='kernel',
                    )
            if self.use_bias:
                self.biases = self.add_weight(
                    shape=(self.output_dim,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name='bias',
                )
        else:
            self.kernel_cluster = self.add_weight(
                shape=(self.embed_dim, self.cluster_num),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel-cluster',
            )
            if self.use_bias:
                self.bias_cluster = self.add_weight(
                    shape=(self.cluster_num,),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='bias-cluster',
                )
            self.embeddings, self.projections = [], []
            if self.use_bias:
                self.biases = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if self.bind_embeddings[i]:
                    self.embeddings.append(None)
                else:
                    self.embeddings.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i], embed_dim),
                        initializer=self.embeddings_initializer,
                        regularizer=self.embeddings_regularizer,
                        constraint=self.embeddings_constraint,
                        name='embeddings-{}'.format(i),
                    ))
                if self.bind_projections[i]:
                    self.projections.append(None)
                else:
                    if embed_dim != self.input_dim or self.force_projection:
                        self.projections.append(self.add_weight(
                            shape=(embed_dim, self.input_dim),
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            name='kernel-{}'.format(i),
                        ))
                    else:
                        self.projections.append(None)
                if self.use_bias:
                    self.biases.append(self.add_weight(
                        shape=(self.cutoffs[i + 1] - self.cutoffs[i],),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        name='bias-{}'.format(i),
                    ))
        super(AdaptiveSoftmax, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.output_dim,)

    def call(self, inputs, **kwargs):
        embeddings = inputs[1:1 + (self.cluster_num + 1)]
        projections = inputs[1 + (self.cluster_num + 1):]
        inputs = inputs[0]
        if self.div_val == 1:
            if self.embed_dim != self.input_dim or self.force_projection:
                projection = self.projections
                if projection is None:
                    projection = projections[0]
                inputs = K.dot(inputs, K.transpose(projection))
            embedding = self.embeddings
            if embedding is None:
                embedding = embeddings[0]
            out = K.dot(inputs, K.transpose(embedding))
            if self.use_bias:
                out = K.bias_add(out, self.biases)
            out = keras.activations.softmax(out, axis=-1)
        else:
            cluster_probs = None
            outputs = []
            for i in range(len(self.cutoffs) - 1):
                embed_dim = self.embed_dim // (self.div_val ** i)
                if embed_dim != self.input_dim or self.force_projection:
                    projection = self.projections[i]
                    if projection is None:
                        projection = projections[i]
                    cluster_input = K.dot(inputs, K.transpose(projection))
                else:
                    cluster_input = inputs
                embedding = self.embeddings[i]
                if embedding is None:
                    embedding = embeddings[i]
                cluster_output = K.dot(cluster_input, K.transpose(embedding))
                if self.use_bias:
                    cluster_output = K.bias_add(cluster_output, self.biases[i])
                if cluster_probs is None:
                    cluster_probs = K.dot(cluster_input, self.kernel_cluster)
                    if self.use_bias:
                        cluster_probs = K.bias_add(cluster_probs, self.bias_cluster)
                    cluster_output = K.concatenate([cluster_output, cluster_probs], axis=-1)
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_probs = cluster_output[..., -self.cluster_num:]
                    cluster_output = cluster_output[..., :-self.cluster_num]
                else:
                    cluster_output = keras.activations.softmax(cluster_output, axis=-1)
                    cluster_output = cluster_output * K.expand_dims(cluster_probs[..., i - 1])
                outputs.append(cluster_output)
            out = K.concatenate(outputs, axis=-1)

        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embed_dim': self.embed_dim,
            'cutoffs': self.cutoffs,
            'div_val': self.div_val,
            'use_bias': self.use_bias,
            'force_projection': self.force_projection,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bind_embeddings': self.bind_embeddings,
            'bind_projections': self.bind_projections,
         }
        base_config = super(AdaptiveSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
