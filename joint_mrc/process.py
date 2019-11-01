# -*- coding: utf-8 -*-
from utils.data_utils import *
from transformer_contrib.keras_bert import load_tokenizer
import numpy as np
import os

class Processor(object):
    """预处理类"""
    def __init__(self, config):
        self.config = config
        self.tokenizer = load_tokenizer(config.vocab_path)

    def _encode(self, q_tokens, c_tokens):
        rest_len = self.config.seq_maxlen - len(q_tokens) - 1
        if len(c_tokens) > rest_len:
            c_tokens = c_tokens[:rest_len]
        c_tokens += [self.tokenizer._token_sep]
        tokens = q_tokens + c_tokens
        pad_len = int(self.config.seq_maxlen - len(tokens))
        seq = self.tokenizer._convert_tokens_to_ids(tokens) + [0] * pad_len
        seg = [0] * len(q_tokens) + [1] * len(c_tokens) + [0] * pad_len
        return seq, seg, len(q_tokens), len(c_tokens)

    def _tokenize(self, text):
        return self.tokenizer._tokenize(text)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def rematch(self, text, tokens):
        return self.tokenizer.rematch(text, tokens)
        
    def process(self, context, question, answer=None, answer_start=None,
                answer_end=None, q_id=None, mode='train'):
        """process raw input data both for training and infering
        
        Arguments:
            context {str} -- 可能包含答案的篇章段落
            question {str} -- 问题文本
        
        Keyword Arguments:
            answer {str} -- 答案文本，推断过程无答案 (default: None)
            answer_start {int} -- 答案在篇章中的起始位置 (default: None)
            answer_end {int} -- 答案在篇章中的结束位置 (default: None)
            mode {str} -- 过程模式，包括["train", "eval", "infer"] (default: 'train')
        
        Returns:
            dict -- 包含字段: {'q_id', 'context', 'inputs', 'outputs'},
                    inputs字段的值为seq和seg序列, outputs字段的值包含nsp，start，end序列.
        """
        # 分段
        sub_contexts, context_starts = split_text(context, maxlen=self.config.seq_maxlen)
        results = []
        q_tokens = self.tokenize(strip_punct(question))  # [[CLS], X, X,..., [SEP]]
        for text, start in zip(sub_contexts, context_starts):
            c_tokens = self._tokenize(text)  # [X, X,..., X] without [CLS] and [SEP]
            result = {'q_id': q_id}
            if mode == 'train':  # train mode needs no context text
                pass
            elif mode in ['infer', 'eval']:
                result['context'] = text
            else:
                raise Exception('mode must be in ["train", "eval", "infer"]!')
            """
            如果没有答案，则包含两种情况：
            1）处理过程为测试的预处理时，不需要outputs字段；
            2）处理过程为训练的预处理时，该篇章为负样本，不包含答案
            """
            if not answer:
                seq, seg, _, _ = self._encode(q_tokens, c_tokens)
                result['inputs'] = [seq, seg]
                result['outputs'] = [0, None, None]
                results.append(result)
                continue

            answer_relative_start = answer_start - start
            if answer_end:
                answer_relative_end = answer_end - start
            else:
                answer_relative_end = answer_relative_start + len(answer)
            if answer_relative_end < 0 or answer_relative_start > len(text): # 答案不在子区间
                seq, seg, _, _ = self._encode(q_tokens, c_tokens)
                result['inputs'] = [seq, seg]
                result['outputs'] = [0, None, None]
                results.append(result)
                continue
            elif answer_relative_start < 0 or answer_relative_end > len(text): # 答案跨子区间
                continue

            
            intervals = self.rematch(text, c_tokens)
            a_start, a_end = self.tokenizer.transform_bound(intervals,
                                                            answer_relative_start,
                                                            answer_relative_end)
            
            seq, seg, _, c_len = self._encode(q_tokens, c_tokens)
            if a_start >= c_len - 1:  # 答案超出序列边界
                result['inputs'] = [seq, seg]
                result['outputs'] = [0, None, None]
                results.append(result)
                continue
            elif a_end >= c_len - 1:  # 答案跨序列边界
                continue
            else:
                result['inputs'] = [seq, seg]
                result['outputs'] = [1, a_start + len(q_tokens), a_end + len(q_tokens)]
                results.append(result)

        return results

    def reduce_answer(self, results):
        for q_id, res_list in results.items():
            score, answer = max(res_list)  
            results[q_id] = {'answer': answer, 'score': score}

def _retrieve(scores, threshold=0.1, top_min=2, top_max=5):
    """第一阶段检索过滤
    
    Arguments:
        scores {list} -- 分数序列
    
    Keyword Arguments:
        threshold {float} -- 检测阈值，降低阈值可以提高答案覆盖率，但增加负样本 (default: {0.1})
        top_min {int} -- 如果检测不到正样本，则直接取最大的top_min个 (default: {2})
        top_max {int} -- 如果检测到正样本，则最多不超过top_max个 (default: {5})
    
    Returns:
        [type] -- [description]
    """
    pred = [i for i, s in enumerate(scores) if s > threshold]
    argsort = np.argsort(-np.asarray(scores))
    if pred:
        if len(pred) <= top_max:
            return pred
        else:
            return argsort[:top_max]
    return argsort[:top_min]
    
                
def process_data(data, config, retrieve=None):
    """预处理数据集
    
    Arguments:
        data {list} -- 样本列表，每一个样本为一个dict
        config {Config} -- 配置对象
    
    Keyword Arguments:
        retrieve {keras model} -- 第一阶段检索模型 (default: {None})
    
    Returns:
        list -- 返回处理后的结果，每条数据为一个dict
    """
    print('Prepare data ...', end='')
    processor = Processor(config)
    processed = []
    for record in data:
        context = record['context']
        for qa in record['qas']:
            question = qa['question']
            answer = qa['answer']
            q_id = qa['q_id']
            answer_start = qa['answer_start']
            answer_end = None if 'answer_end' not in qa or not qa['answer_end'] else qa['answer_end']
            samples = processor.process(context, question, answer,
                                        answer_start, answer_end, q_id=q_id)
            processed += samples
    print('Done!')
    """
    如果没有第一阶段retrieve模型，则直接返回所有处理结果，
    否则用retrieve模型对结果过滤。
    """
    if retrieve is None:
        return processed
    
    seq, seg = [], []
    ans = {} # {'scores': [], 'idxs': [], 'nsps': []}
    for i, d in enumerate(processed):
        inputs, outputs = d['inputs'], d['outputs']
        seq.append(inputs[0])
        seg.append(inputs[1])
        q_id = d['q_id']
        if q_id not in ans:
            ans[q_id] = {'scores': [], 'idxs': [], 'nsps': []}
        ans[q_id]['idxs'].append(i)
        ans[q_id]['nsps'].append(outputs[0])
    
    seq = np.asarray(seq, dtype='int32')
    seg = np.asarray(seg, dtype='int32')
    prob = retrieve.predict([seq, seg], batch_size=32)
    for i, p in enumerate(prob):
        q_id = processed[i]['q_id']
        ans[q_id]['scores'].append(p[0])
        
    filtered, res = [], []
    for q_id, item in ans.items():
        scores, idxs, nsps = item['scores'], item['idxs'], item['nsps']
        if not any(nsps):
            continue
        top = _retrieve(scores)
        pos_idxs = [i for i,nsp in enumerate(nsps) if nsp]
        res.append(bool(set(pos_idxs) & set(top)))
        for idx in top:  # 对结果过滤
            filtered.append(processed[idxs[idx]])
    
    covered = float(sum(res)) / len(res)
    print('Result covers {:.4} answers'.format(covered))
    return filtered


if __name__ == '__main__':
    from utils.io_utils import read_datasets, prepare_data, write_file, read_file, load_mrc
    from utils import keras
    from config import config
    import pickle, time, random

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
     
    train, valid = read_datasets(config.data_dir)
    start = time.time()

    # 查找有没有第一阶段模型
    try:  # 如果有retrieve模型则用其过滤
        retrieve = load_mrc(config.retrieve_dir)
        retrieve = keras.models.Model(retrieve.inputs, retrieve.outputs[0])
        print('Load retrieve done.')
    except:  # 否则不过滤，加载所有的正例负例
        print('No retrieve in path.')
        retrieve = None
    valid_data = process_data(valid, config, retrieve=retrieve)
    train_data = process_data(train, config, retrieve=retrieve)

    with open(config.processed_path, 'wb') as fp:
        pickle.dump((train_data, valid_data), fp)
            
    print('Train samples: {}'.format(len(train_data)))
    print('Valid samples: {}'.format(len(valid_data)))
    pos_num, neg_num = 0, 0
    for data in train_data:
        nsp = data['outputs'][0]
        if nsp:
            pos_num += 1
        else:
            neg_num += 1
    print('Positive samples: {}, Negtive samples: {}'.format(pos_num, neg_num))
    print('Spent {:.2} minuts.'.format((time.time() - start) / 60))
    
    
            
            
            
            
            

        
        