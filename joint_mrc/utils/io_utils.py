# -*- coding: utf-8 -*-
import os, json, logging
from . import keras

__all__ = [
    'read_file', 'read_datasets', 'prepare_data', 'load_ensemble',
    'load_mrc', 'write_file',
]

def read_file(path):
    data = []
    with open(path, 'rb') as fp:
        for line in fp:
            x = json.loads(line.strip().decode('utf8'))
            data.append(x)
    logging.info('Reading file {} done!'.format(os.path.split(path)[-1]))
    return data
    
def read_datasets(path, testing=False):
    dirs = os.listdir(path)
    train, valid, test = [], [], []
    for dir in dirs:
        dir_ = os.path.join(path, dir)
        if os.path.isdir(dir_):
            if testing:
                test += read_file(os.path.join(dir_, 'test.json'))
            else:
                train += read_file(os.path.join(dir_, 'train.json'))
                if os.path.exists(os.path.join(dir_, 'valid.json')):
                    valid += read_file(os.path.join(dir_, 'valid.json'))
            logging.info('Loading dataset {} done!'.format(dir))
    if testing:
        return test
    else:
        return train, valid

def write_file(data, path):
    with open(path, 'wb') as fp:
        for d in data:
            line = json.dumps(d, ensure_ascii=False) + '\n'
            fp.write(line.encode('utf8'))

def prepare_data(data):
    params = []
    for d in data:
        context = d['context']
        qas = d['qas']
        for qa in qas:
            question = qa['question']
            answer = qa['answer']
            answer_start = qa['answer_start']
            if 'q_id' in qa and qa['q_id']:
                q_id = qa['q_id']
            else:
                q_id = len(params)
            params.append(
                {
                    'q_id': q_id,
                    'question': question,
                    'answer': answer,
                    'answer_start': answer_start,
                    'context': context,
                }
            )
    return params
            
def load_ensemble(paths):
    models = [keras.models.load_model(path) for path in paths]
    seq_shape, seg_shape = models[0].input_shape
    seq = keras.layers.Input(shape=seq_shape[1:])
    seg = keras.layers.Input(shape=seg_shape[1:])
    nsps, starts, ends = [], [], []
    for i, model in enumerate(models):
        model.name += '_%d'%i
        nsp_pdf, start_pdf, end_pdf = model([seq, seg])
        nsps.append(nsp_pdf)
        starts.append(start_pdf)
        ends.append(end_pdf)

    nsp = keras.layers.Average()(nsps)
    start = keras.layers.Average()(starts)
    end = keras.layers.Average()(ends)
    ensemble = keras.models.Model([seq, seg], [nsp, start, end])
    logging.info('Loading ensemble model Done!')
    return ensemble

def load_mrc(path, ensemble=False):
    if os.path.isdir(path):
        files = filter(lambda x: '.hdf5' in x or '.h5' in x, os.listdir(path))
        files = [os.path.join(path, f) for f in files]
        if not files:
            raise Exception('There is no keras model in {}'.format(path))
        if ensemble:
            model = load_ensemble(files)
        else:
            model = keras.models.load_model(files[0])
            logging.info('Loading model {} done!'.format(os.path.split(files[0])[-1]))
    else:
        if not path.endswith('.hdf5') and not path.endswith('.h5'):
            raise Exception('{} is not a keras model file!'.format(path))
        model = keras.models.load_model(path)
        logging.info('Loading model {} done!'.format(os.path.split(path)[-1]))
    return model
