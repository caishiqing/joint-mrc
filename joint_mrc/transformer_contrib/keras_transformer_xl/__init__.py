from .identity import *
from .pos_embed import *
from .rel_multi_head import *
from .rel_bias import *
from .memory import *
from .scale import *
from .transformer_xl import *
from .loader import *
from .sequence import *
from .backend import utils

utils.get_custom_objects().update(
    {
        'Identity' : Identity,
        'Memory' : Memory,
        'PositionalEmbedding' : PositionalEmbedding,
        'RelativeBias' : RelativeBias,
        'RelativePartialMultiHeadSelfAttention' : RelativePartialMultiHeadSelfAttention,
        'Scale' : Scale,
        'MemorySequence' : MemorySequence,
        }
    )
