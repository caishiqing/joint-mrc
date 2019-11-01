import json
import tensorflow as tf
from .model import get_gpt2
from .bpe import get_bpe_from_files


__all__ = ['load_trained_model_from_checkpoint',
           'load_gpt2_from_ckpt',
           'load_bpe_from_ckpt']


def load_trained_model_from_checkpoint(config_path,
                                       checkpoint_path,
                                       seq_len=None,
                                       batch_size=None,
                                       fixed_input_shape=False,
                                       return_last_layer=True,
                                       compile=False):
    """Load trained official model from checkpoint.

    :param config_path: The path to the JSON configuration file. (hparams.json)
    :param checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :param batch_size: Batch size of the model.
    :param fixed_input_shape: Whether the length of input is fixed. (Needed for TPU training)
    :param return_last_layer: Wether return output layer.
    :return: The model.
    """
    with open(config_path, 'r') as reader:
        config = json.load(reader)
    if seq_len is None:
        n_ctx = config['n_ctx']
    else:
        n_ctx = min(seq_len, config['n_ctx'])
    n_embd = config['n_embd']
    model = get_gpt2(
        n_vocab=config['n_vocab'],
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        batch_size=batch_size,
        fixed_input_shape=fixed_input_shape,
        return_last_layer=return_last_layer,
        compile=compile,
    )
    model.get_layer(name='Embed-Token').set_weights([
        tf.train.load_variable(checkpoint_path, 'model/wte:0'),
    ])
    model.get_layer(name='Embed-Token-Pos').set_weights([
        tf.train.load_variable(checkpoint_path, 'model/wpe:0')[:seq_len, :],
    ])
    for i in range(config['n_layer']):
        model.get_layer(name='Encode-%d-MultiHeadAtt-Norm' % i).set_weights([
            tf.train.load_variable(checkpoint_path, 'model/h%d/ln_1/g:0' % i),
            tf.train.load_variable(checkpoint_path, 'model/h%d/ln_1/b:0' % i),
        ])
        kernel = tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_attn/w:0' % i)[0]
        bias = tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_attn/b:0' % i)
        model.get_layer(name='Encode-%d-MultiHeadAtt' % i).set_weights([
            kernel[:, :n_embd],
            bias[:n_embd],
            kernel[:, n_embd:-n_embd],
            bias[n_embd:-n_embd],
            kernel[:, -n_embd:],
            bias[-n_embd:],
            tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_proj/w:0' % i)[0],
            tf.train.load_variable(checkpoint_path, 'model/h%d/attn/c_proj/b:0' % i),
        ])
        model.get_layer(name='Encode-%d-FeedForward-Norm' % i).set_weights([
            tf.train.load_variable(checkpoint_path, 'model/h%d/ln_2/g:0' % i),
            tf.train.load_variable(checkpoint_path, 'model/h%d/ln_2/b:0' % i),
        ])
        model.get_layer(name='Encode-%d-FeedForward' % i).set_weights([
            tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_fc/w:0' % i)[0],
            tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_fc/b:0' % i),
            tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_proj/w:0' % i)[0],
            tf.train.load_variable(checkpoint_path, 'model/h%d/mlp/c_proj/b:0' % i),
        ])
    model.get_layer(name='Norm').set_weights([
        tf.train.load_variable(checkpoint_path, 'model/ln_f/g:0'),
        tf.train.load_variable(checkpoint_path, 'model/ln_f/b:0'),
    ])
    return model

def load_gpt2_from_ckpt(ckpt_path,
                        seq_len=None,
                        batch_size=None,
                        fixed_input_shape=False,
                        return_last_layer=True,
                        compile=False):
    import os
    config_path = os.path.join(ckpt_path, 'hparams.json')
    checkpoint_path = os.path.join(ckpt_path, 'model.ckpt')
    return load_trained_model_from_checkpoint(config_path,
                                              checkpoint_path,
                                              seq_len,
                                              batch_size,
                                              fixed_input_shape,
                                              return_last_layer,
                                              compile)

def load_bpe_from_ckpt(ckpt_path):
    import os
    encoder_path = os.path.join(ckpt_path, 'encoder.json')
    vocab_path = os.path.join(ckpt_path, 'vocab.bpe')
    return get_bpe_from_files(encoder_path, vocab_path)
