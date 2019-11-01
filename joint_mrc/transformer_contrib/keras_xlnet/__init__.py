from .permutation import *
from .mask_embed import *
from .position_embed import *
from .segment_bias import *
from .segment_embed import *
from .attention import *
from .xlnet import *
from .loader import *
from .tokenizer import *
from .pretrained import *
from .backend import utils

utils.get_custom_objects().update(
    {
        'PermutationMask': PermutationMask,
        'MaskEmbedding': MaskEmbedding,
        'PositionalEmbedding': PositionalEmbedding,
        'SegmentBias': SegmentBias,
        'RelativeSegmentEmbedding': RelativeSegmentEmbedding,
        'RelativePartialMultiHeadSelfAttention': RelativePartialMultiHeadSelfAttention,
        }
    )
