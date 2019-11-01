from .backend import keras
from ..keras_layers import (EmbeddingRet, EmbeddingSim,
                            PositionEmbedding, LayerNormalization)
from ..keras_transformer import gelu, attention_builder, feed_forward_builder


def _wrap_layer(name, input_layer, build_func, trainable=True):
    """Wrap layers with normalization and residual.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    build_output = build_func(normal_layer)
    return keras.layers.Add(name='%s-Add' % name)([input_layer, build_output])


def _get_encoder_component(name,
                           input_layer,
                           head_num,
                           hidden_dim,
                           attention_activation=None,
                           feed_forward_activation='relu',
                           trainable=True):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadAtt' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    return feed_forward_layer


def get_gpt2(n_vocab,
             n_ctx=1024,
             n_embd=768,
             n_head=12,
             n_layer=12,
             hidden_dim=None,
             batch_size=None,
             fixed_input_shape=False,
             return_last_layer=True,
             compile=True):
    """Get basic GPT-2 model.

    :param n_vocab: Number of vocabulary tokens.
    :param n_ctx: The length of each input.
    :param n_embd: The dimension of embeddings.
    :param n_head: Number of heads in transformer.
    :param hidden_dim: The dimension of hidden layer.
    :param n_layer: Number of transformer blocks.
    :param batch_size: Batch size of the model.
    :param fixed_input_shape: Whether the length of input is fixed. (Needed for TPU training)
    :param return_last_layer: Wether return output layer.
    :return: The model.
    """
    if fixed_input_shape:
        input_layer_shape = (batch_size, n_ctx)
    else:
        input_layer_shape = (batch_size, None)
    input_layer = keras.layers.Input(
        batch_shape=input_layer_shape,
        name='Input',
    )

    embed_token, embeddings = EmbeddingRet(
        input_dim=n_vocab,
        output_dim=n_embd,
        mask_zero=True,
        name='Embed-Token',
    )(input_layer)
    embed_token_pos = PositionEmbedding(
        input_dim=n_ctx,
        output_dim=n_embd,
        mode=PositionEmbedding.MODE_ADD,
        name='Embed-Token-Pos',
    )(embed_token)

    if not hidden_dim:
        hidden_dim = n_embd * 4
    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = _get_encoder_component(
            name='Encode-%d' % i,
            input_layer=last_layer,
            head_num=n_head,
            hidden_dim=hidden_dim,
            attention_activation=None,
            feed_forward_activation=gelu,
        )

    norm_layer = LayerNormalization(
        name='Norm',
    )(last_layer)

    if not return_last_layer:
        model = keras.models.Model(inputs=input_layer, outputs=norm_layer)
        return model
    
    output_layer = EmbeddingSim(
        use_bias=False,
        name='Output',
    )([norm_layer, embeddings])

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    if compile:
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
    return model

