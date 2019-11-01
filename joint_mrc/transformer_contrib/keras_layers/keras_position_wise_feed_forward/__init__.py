from .feed_forward import FeedForward
from .backend import utils

utils.get_custom_objects().update(
    {
        'FeedForward' : FeedForward,
        }
    )
