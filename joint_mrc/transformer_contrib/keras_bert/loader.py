import json,os
import numpy as np
import tensorflow as tf
from .bert import get_bert
from .tokenizer import Tokenizer, load_vocab
from .backend import keras


__all__ = [
    'build_model_from_config',
    'load_model_weights_from_checkpoint',
    'load_trained_model_from_checkpoint',
    'load_bert_from_ckpt',
    'load_bert_from_hdf5',
    'load_tokenizer',
]


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def build_model_from_config(config_file,
                            dropout=None,
                            transformer_num=None,
                            training=False,
                            trainable=None,
                            seq_len=None):
    """Build the model from config file.

    :param config_file: The path to the JSON configuration file.
    :param training: If training, the whole model will be returned.
    :param trainable: Whether the model is trainable.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model and config
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = min(seq_len, config['max_position_embeddings'])
    if transformer_num is not None:
        config['num_hidden_layers'] = transformer_num
    if dropout is not None:
        config['attention_probs_dropout_prob'] = dropout
        config['hidden_dropout_prob'] = dropout
    if trainable is None:
        trainable = training

    model = get_bert(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=config['max_position_embeddings'],
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        training=training,
        trainable=trainable,
    )
    if not training:
        inputs, outputs = model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
    return model, config


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=False):
    """Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: Loaded configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
        ])
    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])


def load_trained_model_from_checkpoint(config_file,
                                       checkpoint_file,
                                       dropout=None,
                                       transformer_num=None,
                                       training=False,
                                       trainable=None,
                                       seq_len=None):
    """Load trained official model from checkpoint.

    :param config_file: The path to the JSON configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    :param trainable: Whether the model is trainable. The default value is the same with `training`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model
    """
    model, config = build_model_from_config(config_file, dropout=dropout,
                                            transformer_num=transformer_num,
                                            training=training, trainable=trainable,
                                            seq_len=seq_len)
    load_model_weights_from_checkpoint(model, config, checkpoint_file, training=training)
    return model

def load_bert_from_ckpt(ckpt_path, **kwargs):
    config_file = os.path.join(ckpt_path, 'bert_config.json')
    checkpoint_file = os.path.join(ckpt_path, 'bert_model.ckpt')
    return load_trained_model_from_checkpoint(config_file, checkpoint_file,
                                              **kwargs)

def load_bert_from_hdf5(hdf5_file,
                        transformer_num=None,
                        trainable=False,
                        training=False,
                        seq_len=None,
                        dropout=None):
    """load trained bert model from hdf5 file
    """
    bert = keras.models.load_model(hdf5_file)
    if dropout is None and transformer_num is None and seq_len is None:
        bert.trainable = trainable
        return bert

    vocab_size = bert.get_layer(name='Embedding-Token').input_dim
    if transformer_num is None:
        transformer_num = 12
    if dropout is None:
        dropout = 0.1

    inputs, outputs = get_bert(vocab_size, pos_num=seq_len, seq_len=seq_len,
                               transformer_num=transformer_num,
                               dropout_rate=dropout, training=training,
                               trainable=trainable)

    model = keras.models.Model(inputs, outputs)
    
    model.get_layer(name='Embedding-Token').set_weights(
        bert.get_layer(name='Embedding-Token').get_weights(),
    )
    model.get_layer(name='Embedding-Position').set_weights([
        bert.get_layer(name='Embedding-Position').get_weights()[0][:seq_len, :],
    ])
    model.get_layer(name='Embedding-Segment').set_weights(
        bert.get_layer(name='Embedding-Segment').get_weights()
    )
    model.get_layer(name='Embedding-Norm').set_weights(
        bert.get_layer(name='Embedding-Norm').get_weights()
    )
    for i in range(transformer_num):
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights(
            bert.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).get_weights()
        )
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights(
            bert.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).get_weights()
        )
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights(
            bert.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).get_weights()
        )
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights(
            bert.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).get_weights()
        )
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights(
            bert.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).get_weights()
        )
    if training:
        model.get_layer(name='MLM-Dense').set_weights(
            bert.get_layer(name='MLM-Dense').get_weights()
        )
        model.get_layer(name='MLM-Norm').set_weights(
            bert.get_layer(name='MLM-Norm').get_weights()
        )
        model.get_layer(name='MLM-Sim').set_weights(
            bert.get_layer(name='MLM-Sim').get_weights()
        )
        model.get_layer(name='NSP-Dense').set_weights(
            bert.get_layer(name='NSP-Dense').get_weights()
        )
        model.get_layer(name='NSP').set_weights(
            bert.get_layer(name='NSP').get_weights()
        )
    model.trainable = trainable
    return model


def load_tokenizer(path):
    token_dict = load_vocab(path)
    tokenizer = Tokenizer(token_dict)
    return tokenizer
    
    

