from .seq_self_attention import SeqSelfAttention
from .seq_weighted_attention import SeqWeightedAttention
from .scaled_dot_attention import ScaledDotProductAttention
from .backend import utils

utils.get_custom_objects().update(
    {
        'SeqSelfAttention' : SeqSelfAttention,
        'SeqWeightedAttention' : SeqWeightedAttention,
        'ScaledDotProductAttention' : ScaledDotProductAttention,
        }
    )
