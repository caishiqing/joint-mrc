from .multi_head import MultiHead
from .multi_head_attention import MultiHeadAttention
from .backend import utils

utils.get_custom_objects().update(
    {
        'MultiHead': MultiHead,
        'MultiHeadAttention': MultiHeadAttention
        }
)
