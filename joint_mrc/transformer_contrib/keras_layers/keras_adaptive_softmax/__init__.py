from .embedding import *
from .softmax import *
from .backend import utils

utils.get_custom_objects().update(
    {
        'AdaptiveEmbedding' : AdaptiveEmbedding,
        'AdaptiveSoftmax' : AdaptiveSoftmax
        }
    )
