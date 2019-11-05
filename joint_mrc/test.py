from utils.io_utils import load_mrc
from model import JointMRC
import tensorflow as tf
from config import config

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
    model = load_mrc(config.model_dir)
    try:
        retrieve = load_mrc(config.retrieve_dir)
    except:
        retrieve = None
    # load model
    mrc = JointMRC(config, model, retrieve)
    
    q = '姚明有多高？'
    c1 = '姚明身高226cm，被称为小巨人。'
    c2 = '姚明的父亲叫姚志源，身高208cm'
    params = [
        {'q_id': '1', 'question': q, 'context': c1},
        {'q_id': '1', 'question': q, 'context': c2},
        ]
    
    result = mrc.infer(params)
    print(result)
    
    
    
    