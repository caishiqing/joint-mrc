# -*- coding: utf-8 -*-
import os
from model import  JointMRC
from utils.train_utils import *
from utils import io_utils, eval_utils, keras
from utils import backend as K
from collections import OrderedDict
import tensorflow as tf
import pickle, argparse
import numpy as np
from config import config


def parse_args():
    """命令行参数解析
    Args:
        --action: 执行动作类型，包括'train'和'eval'
        --ensemble: 是否使用集成
        --k_fold: 如果训练集成，将数据切割成多少份
        --gpu: 使用的GPU序号
        --batch_size: 每种动作对应的batch大小
        --model_path: 存储（训练）或者加载（评估）模型路径
        --file_path: 评估文件路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default='train', help='mode of ["train", "eval"]')
    parser.add_argument("--ensemble", action='store_true', default=False)
    parser.add_argument("--k_fold", type=int, default=5, help='k-fold split number')
    parser.add_argument("--gpu", type=str, default='', help='chose gpu')
    parser.add_argument("--batch_size", type=int, default=0, help='batch size')
    parser.add_argument("--model_path", type=str, default='', help='model file path')
    parser.add_argument("--file_path", type=str, default='', help='target file path')
    args = parser.parse_args()
    return args


def train(config, args):
    """训练主程序

    如果args指定model_path则模型参数存储到model_path文件中，否则按命名规则存储到
    config的model_dir目录下；如果指定ensemble，则必须指定k_fold参数，并且按
    k_fold多次划分数据并训练k_fold个模型.
    
    Arguments:
        config {Config} -- 文件配置参数，参见config.py
        args {ArgumentParser} -- 命令行参数，如果与config有重复参数，则args覆盖config参数
    
    Returns:
        MRC实例 -- 返回MRC模型实例并存储模型参数到指定目录
    """
    gpu_num = len(args.gpu.split(','))
    args.__setattr__('gpu_num', gpu_num)
    if hasattr(args, 'batch_size') and args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = config.batch_size

    if gpu_num > 1:
        if config.first_gpu:
            batch_size *= gpu_num
        else:
            batch_size *= (gpu_num - 1)

    with open(config.processed_path, 'rb') as fp:
        train_data, valid_data = pickle.load(fp)
    
    # 按照k-fold重新划分数据
    if args.ensemble:
        assert(hasattr(args, 'k_fold') and args.k_fold)
        data = train_data + valid_data
        splits = split_data(data[:], args.k_fold, True)
        del data, train_data, valid_data
        for k in range(args.k_fold):
            valid_data = splits[k][:]
            train_data = []
            for n, split in enumerate(splits):
                if n != k:
                    train_data += split[:]
            train_loops = compute_loops(train_data, batch_size, config.positive_first)
            valid_loops = compute_loops(valid_data, batch_size, config.positive_first)
            train_gen = generate_data(train_data, batch_size, True, config.positive_first)
            valid_gen = generate_data(valid_data, batch_size, True, config.positive_first)
            args.__setattr__('train_loops', train_loops)
            args.__setattr__('valid_loops', valid_loops)
            mrc = JointMRC(config=config)
            mrc.train(train_gen, valid_gen, args)
            K.clear_session()
        return

    train_loops = compute_loops(train_data, batch_size, config.positive_first)
    valid_loops = compute_loops(valid_data, batch_size, config.positive_first)
    train_gen = generate_data(train_data, batch_size, True, config.positive_first)
    valid_gen = generate_data(valid_data, batch_size, True, config.positive_first)
    args.__setattr__('train_loops', train_loops)
    args.__setattr__('valid_loops', valid_loops)  
    mrc = JointMRC(config=config)
    mrc.train(train_gen, valid_gen, args)
    return mrc


def eval(config, args):
    """评估主程序

    如果args指定model_path则从model_path文件中加载模型参数，否则到
    config的model_dir目录下查找第一个后缀为.hdf5的文件加载参数;
    如果model_path为目录并且指定ensemble参数，则集成model_path中所有模型

    如果args指定file_path则对file_path文件中的数据做评估，否则到config
    的data_dir目录下读取所有数据集的test.json文件中的数据
    
    Arguments:
        config {Config} -- 文件配置参数，参见config.py
        args {ArgumentParser} -- 命令行参数，如果与config有重复参数，则args覆盖config参数
    
    Returns:
        dict -- 返回各个指标结果字典
    """
    if hasattr(args, 'model_path') and args.model_path:
        model_path = args.model_path
    else:
        files = os.listdir(config.model_dir)
        file = list(filter(lambda x: '.hdf5' in x, files))[0]
        model_path = os.path.join(config.model_dir, file)
    if hasattr(args, 'batch_size') and args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = config.batch_size
    if hasattr(args, 'file_path') and args.file_path:
        test_data = io_utils.read_file(args.file_path)
    else:
        test_data = io_utils.read_datasets(config.data_dir, testing=True)
    model = io_utils.load_mrc(model_path, args.ensemble)
    try:
        retrieve = io_utils.load_mrc(config.retrieve_dir)
        print('Load retrieve done.')
    except:
        print('No retrieve in path.')
        retrieve = None
    mrc = JointMRC(config, model, retrieve)
    
    answers = OrderedDict()
    tests = []
    for test in test_data:
        context = test['context']
        qas = test['qas']
        for qa in qas:
            answer = qa['answer']
            question = qa['question']
            answer = qa['answer']
            answer_start = qa['answer_start']
            if 'q_id' in qa and qa['q_id']:
                q_id = qa['q_id']
            else:
                q_id = '%d' % (len(tests))
            if answer:
                answers[q_id] = answer
            param = {
                'context': context,
                'question': question,
                'q_id': q_id,
                'answer': answer,
                'answer_start': answer_start,
            }
            tests.append(param)

    result = mrc.evaluate(tests, batch_size=batch_size)
    print(result)
    return result

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if args.action=='train':
        train(config, args)
    elif args.action == 'eval':
        eval(config, args)
        
    
    
            
            
