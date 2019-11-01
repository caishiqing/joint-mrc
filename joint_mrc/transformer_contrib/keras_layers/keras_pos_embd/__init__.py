from .pos_embd import PositionEmbedding
from .trig_pos_embd import TrigPosEmbedding
from .backend import utils

utils.get_custom_objects().update(
    {
        'PositionEmbedding' : PositionEmbedding,
        'TrigPosEmbedding' : TrigPosEmbedding,
        }
    )
