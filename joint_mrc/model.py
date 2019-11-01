# -*- coding: utf-8 -*-
from utils import keras, utils, eval_utils, AdamWarmup
from utils import multi_gpu_utils
import tensorflow as tf
from utils.layer_utils import MaskQuest, Pointer
from transformer_contrib.keras_bert import load_bert_from_ckpt
from process import Processor, _retrieve
import numpy as np
from scipy import sparse
from collections import OrderedDict


def get_mrc_model(bert, config=None):
    if config is None or not hasattr(config, 'dropout'):
        dropout = 0.0
    else:
        dropout = config.dropout
    seq, seg = bert.input
    hidden = bert.output
    h_cls = keras.layers.Lambda(lambda x: x[:, 0, :])(hidden)
    h_cls = keras.layers.Dropout(dropout)(h_cls)
    nsp = keras.layers.Dense(1, activation='sigmoid', name='nsp')(h_cls)
    
    h_context = MaskQuest()([hidden, seg])
    h_context = keras.layers.Dropout(dropout)(h_context)
    start = Pointer(mode='categorical', name='start')(h_context)
    end = Pointer(mode='categorical', name='end')(h_context)
    model = keras.models.Model([seq, seg], [nsp, start, end])
    return model


class Answer(keras.layers.Layer):
    """问答解码层"""
    def __init__(self, ans_maxlen, **kwargs):
        super().__init__(**kwargs)
        self.ans_maxlen = ans_maxlen
        
    def call(self, inputs):
        start, end = inputs
        start = tf.expand_dims(start, 2)
        end = tf.expand_dims(end, 1)
        mat = start * end
        mat = tf.matrix_band_part(mat, 0, self.ans_maxlen - 1)
        mat = keras.layers.Flatten()(mat)
        y = tf.argmax(mat, axis=1, output_type='int32')
        return y

    def compute_output_shape(self, input_shape):
        B, _ = input_shape[0]
        return (B,)


class JointMRC(object):

    def __init__(self, config, model=None, retrieve=None):
        self.config = config
        self.processor = Processor(config)
        if model is None:
            bert = load_bert_from_ckpt(self.config.bert_dir,
                                       transformer_num=self.config.n_layers,
                                       trainable=True)
            self.model = get_mrc_model(bert, self.config)
        else:
            self.model = model

        if retrieve is not None:
            self.retrieve = keras.models.Model(retrieve.inputs, retrieve.outputs[0])
        else:
            self.retrieve = None

        start = keras.layers.Input(shape=(config.seq_maxlen,), dtype='float32')
        end = keras.layers.Input(shape=(config.seq_maxlen,), dtype='float32')
        decoded = Answer(config.ans_maxlen)([start, end])
        self.decoder = keras.models.Model([start, end], decoded)

    def _decode(self, start_pdf, end_pdf, segment, batch_size=8):
        """输出解码
        
        Arguments:
            start_pdf {np.ndarray} -- start indice pdf
            end_pdf {np.ndarray} -- end indice pdf
            segment {np.ndarray} -- segment encoding
            batch_size {int} -- batch size if decoding
        
        Returns:
            list -- sample list of (start, end, score)
        """
        n_samples, length = start_pdf.shape
        start_pdf = np.sqrt(start_pdf, dtype='float32')
        end_pdf = np.sqrt(end_pdf, dtype='float32')
        out = []
        decoded = self.decoder.predict([start_pdf, end_pdf], batch_size=batch_size)
        for i in range(n_samples):
            seg, y = segment[i], decoded[i]
            q_len = 0
            for encode in seg:
                if encode == 0:
                    q_len += 1
                else:
                    break
            start, end = int(y / length), int(y % length)
            out.append((start - q_len, end - q_len, start_pdf[i][start] * end_pdf[i][end]))
        return out

    def train(self, train_gen, valid_gen, args):
        import types, os
        from utils.train_utils import save, Checkpoint, LossHistory
            
        self.model.save = types.MethodType(save, self.model)
        self.model.summary()
        # 根据命令行参数gpu判断gpu数量，如果多块gpu则使用多卡并行训练
        if args.gpu_num <= 1:
            model = self.model
        else:
            model = multi_gpu_utils.multi_gpu_model(self.model, gpus=args.gpu_num,
                                                    first_gpu=self.config.first_gpu)

        #optimizer = keras.optimizers.Adam(self.config.lr)
        optimizer = AdamWarmup(
            warmup_steps=args.train_loops, 
            decay_steps=args.train_loops*(self.config.epochs-1),
            learning_rate=self.config.lr,
            min_lr=1e-6,
            )
        model.compile(optimizer=optimizer,
                      loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                      metrics={'nsp': 'accuracy', 'start': 'accuracy', 'end': 'accuracy'},
                      loss_weights=self.config.loss_weights)

        if hasattr(args, 'model_path') and args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(self.config.model_dir, '{}_best.hdf5'.format(self.config.pre_model))
        checkpoint = Checkpoint(filepath=model_path,
                                raw_model=self.model,
                                monitor=self.config.monitor,
                                save_best_only=True)

        model.fit_generator(train_gen,
                            steps_per_epoch=args.train_loops,
                            epochs=self.config.epochs,
                            validation_data=valid_gen,
                            validation_steps=args.valid_loops,
                            callbacks=[checkpoint])

        if hasattr(args, 'model_path') and args.model_path:
            model_file = args.model_path
        else:
            model_file = 'MRC_{}_{}:{:.4}.hdf5'.format(self.config.pre_model, self.config.monitor, checkpoint.best)
            os.rename(model_path, os.path.join(self.config.model_dir, model_file))
        self.model.load_weights(os.path.join(self.config.model_dir, model_file))


    def infer(self, params, batch_size=8):
        """推断接口
        
        Arguments:
            params {list} -- 输入数据，每条数据为一个dict，包含'context'，'question'和
                             'q_id'三个字段，若没有'q_id'则生成一个唯一的'q_id'
        
        Keyword Arguments:
            batch_size {int} -- 推断过程的批大小 (default: {8})
        
        Returns:
            dict -- 结果有序字典{q_id:{'answer':xxx, 'score':xx}, ...}
        """
        datas = []
        seq, seg = [], []
        for i, param in enumerate(params):
            context = param['context']
            question = param['question']
            if 'q_id' in param and param['q_id']:
                q_id = param['q_id']
            else:
                q_id = '%d' % i
            samples = self.processor.process(context, question, mode='infer')
            for sample in samples:
                sample['q_id'] = q_id
                seq.append(sample['inputs'][0])
                seg.append(sample['inputs'][1])
                datas.append(sample)
        
        seq = np.asarray(seq, dtype='int32')
        seg = np.asarray(seg, dtype='int32')

        w1, w2 = self.config.fuse_weights
        epsilon = 1e-3

        # 先用检索模型过滤篇章
        if self.retrieve is not None:  
            filtered_indices = []
            ans = {}
            nsp = self.retrieve.predict([seq, seg], batch_size=batch_size)
            for i, p in enumerate(nsp):
                q_id = datas[i]['q_id']
                if q_id not in ans:
                    ans[q_id] = {'scores': [], 'idxs': []}
                ans[q_id]['scores'].append(p[0])
                ans[q_id]['idxs'].append(i)
            for q_id, item in ans.items():
                scores, idxs = item['scores'], item['idxs']
                top = _retrieve(scores)
                for idx in top:
                    filtered_indices.append(idxs[idx])
            filtered_indices = sorted(filtered_indices)
            seq = seq[filtered_indices]
            seg = seg[filtered_indices]
            datas = [datas[i] for i in filtered_indices]
            
        nsp_prob, start_pdf, end_pdf = self.model.predict([seq, seg], batch_size=batch_size)
        outputs = self._decode(start_pdf, end_pdf, seg, batch_size)

        results = dict()
        assert (len(datas) == len(outputs))
        for sample, output, prob in zip(datas, outputs, nsp_prob):
            text, q_id = sample['context'], sample['q_id']
            token_start, token_end, answer_score = output
            tokens = self.processor._tokenize(text)
            intervals = self.processor.rematch(text, tokens)
            try:
                start = intervals[token_start][0]
                end = intervals[token_end][-1]
            except:
                start = 0
                end = len(text)
            answer = text[start : end]
            score = np.exp((w1 * np.log(prob[0] + epsilon) + w2 * np.log(answer_score + epsilon)) / (w1 + w2))
            
            if q_id not in results:
                results[q_id] = []
            results[q_id].append((float(score), str(answer)))
        self.processor.reduce_answer(results)
        return results

    def evaluate(self, params, batch_size=8):
        data = []
        for param in params:
            data.append(
                {
                    'context': param['context'],
                    'question': param['question'],
                    'q_id': param['q_id'],
                }
            )
        results = self.infer(data, batch_size)

        answers, preds = [], []
        for param in params:
            q_id = param['q_id']
            answer = param['answer']
            if not answer:
                continue
            pred = results[q_id]['answer']
            answers.append(answer)
            preds.append(pred)

        em = eval_utils.evaluate(eval_utils.exact_match, preds, answers)
        f1 = eval_utils.evaluate(eval_utils.F1_score, preds, answers)
        rouge = eval_utils.evaluate(eval_utils.Rouge_L, preds, answers)
        return {'EM': em, 'F1': f1, 'Rouge-L': rouge}


        
        
        
    


            
            
            

             
            
        