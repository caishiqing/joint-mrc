from . import keras
import numpy as np
import random

__all__ = [
    'split_data', 'generate_data', 'save', 'lr_sheduler', 'compute_loops',
    'Checkpoint', 'LossHistory',
]

def split_data(data, k_fold=10, shuffle=True):
    """ k-fold spliting data """
    totle = len(data)
    if shuffle:
        random.shuffle(data)
    batch_size = int(totle / k_fold)
    result = []
    for k in range(k_fold):
        batch = data[k * batch_size : min((k + 1) * batch_size, totle)]
        result.append(batch)
    rest = totle % k_fold
    result[-1] += data[-rest:]
    return result

def generate_data(data, batch_size, shuffle=True, positive_first=True):
    """ dynamically generate data with balanced sampling """
    while True:
        if shuffle:
            random.shuffle(data)
        positives, negtives = [], []
        for i, d in enumerate(data):
            nsp, _, _ = d['outputs']
            if nsp == 1:
                positives.append(i)
            else:
                negtives.append(i)
        indexs = []
        if positive_first:
            while positives:
                p = positives.pop()
                n = random.choice(negtives)#negtives.pop()
                indexs.append(p)
                indexs.append(n)
        else:
            while positives and negtives:
                p = positives.pop()
                n = negtives.pop()
                indexs.append(p)
                indexs.append(n)

        totle = len(indexs)
        loops = int(totle / batch_size)
        
        for loop in range(loops):
            sub_indexs=indexs[loop * batch_size : min((loop + 1) * batch_size, totle)]
            seq, seg, nsp, start, end = [], [], [], [], []
            for idx in sub_indexs:
                d = data[idx]
                inputs, outputs = d['inputs'], d['outputs']
                seq.append(inputs[0])
                seg.append(inputs[1])
                nsp.append(outputs[0])
                start_y = [0] * len(inputs[0])
                end_y = start_y[:]
                if outputs[0]:
                    start_y[outputs[1]] = 1
                    end_y[outputs[2]] = 1
                start.append(start_y)
                end.append(end_y)
            seq = np.asarray(seq, dtype='int32')
            seg = np.asarray(seg, dtype='int32')
            nsp = np.asarray(nsp, dtype='float32')
            start = np.asarray(start, dtype='float32')
            end = np.asarray(end, dtype='float32')
            yield [seq, seg], [nsp, start, end]
            
def save(cls, filepath, overwrite=True):
    """ save model without optimizer states """
    keras.models.save_model(cls, filepath, overwrite=overwrite, include_optimizer=False)

def lr_sheduler(decay=0.9, min_rate=1e-6):
    def lrshedular(epoch, lr):
        return max(lr * (decay ** epoch), min_rate)
    return lrshedular

def compute_loops(data, batch_size, positive_first=True):
    """ compute steps in one epoch with balanced sampling """
    pos_num, neg_num = 0, 0
    if positive_first:
        for d in data:
            nsp = d['outputs'][0]
            if nsp == 1:
                pos_num += 1
        totle = 2 * pos_num
    else:
        for d in data:
            nsp = d['outputs'][0]
            if nsp == 1:
                pos_num += 1
            else:
                neg_num += 1
        totle = 2 * min(pos_num, neg_num)
    loops = int(totle / batch_size)
    return loops
    


class Checkpoint(keras.callbacks.ModelCheckpoint):
    """ checkpoint callback with raw(single) model """
    def __init__(self, filepath, raw_model, **kwargs):
        if 'loss' in kwargs['monitor']:
            kwargs['mode'] = 'min'
        elif 'acc' in kwargs['monitor']:
            kwargs['mode'] = 'max'
        else:
            kwargs['mode'] = 'auto'
        super(Checkpoint, self).__init__(filepath, **kwargs)
        self.raw_model = raw_model
        
    def set_model(self, model):
        super(Checkpoint, self).set_model(self.raw_model)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.nsp_loss = {'batch': [], 'epoch': []}
        self.start_loss = {'batch': [], 'epoch': []}
        self.end_loss = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_nsp_loss = {'batch': [], 'epoch': []}
        self.val_start_loss = {'batch': [], 'epoch': []}
        self.val_end_loss = {'batch': [], 'epoch': []}
        
        self.nsp_acc = {'batch': [], 'epoch': []}
        self.start_acc = {'batch': [], 'epoch': []}
        self.end_acc = {'batch': [], 'epoch': []}
        self.val_nsp_acc = {'batch': [], 'epoch': []}
        self.val_start_acc = {'batch': [], 'epoch': []}
        self.val_end_acc = {'batch': [], 'epoch': []}
    

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.nsp_loss['batch'].append(logs.get('nsp_loss'))
        self.start_loss['batch'].append(logs.get('start_loss'))
        self.end_loss['batch'].append(logs.get('end_loss'))
        
        self.nsp_acc['batch'].append(logs.get('nsp_acc'))
        self.start_acc['batch'].append(logs.get('start_acc'))
        self.end_acc['batch'].append(logs.get('end_acc'))
        

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.nsp_loss['epoch'].append(logs.get('nsp_loss'))
        self.start_loss['epoch'].append(logs.get('start_loss'))
        self.end_loss['epoch'].append(logs.get('end_loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_nsp_loss['epoch'].append(logs.get('val_nsp_loss'))
        self.val_start_loss['epoch'].append(logs.get('val_start_loss'))
        self.val_end_loss['epoch'].append(logs.get('val_end_loss'))
        
        self.nsp_acc['epoch'].append(logs.get('nsp_acc'))
        self.start_acc['epoch'].append(logs.get('start_acc'))
        self.end_acc['epoch'].append(logs.get('end_acc'))
        self.val_nsp_acc['epoch'].append(logs.get('val_nsp_acc'))
        self.val_start_acc['epoch'].append(logs.get('val_start_acc'))
        self.val_end_acc['epoch'].append(logs.get('val_end_acc'))
        

    def plot(self, path='train.png'):
        from matplotlib import pyplot as plt
        iters = range(len(self.nsp_loss['batch']))
        epochs = range(len(self.nsp_loss['epoch']))

        plt.figure(figsize=(12, 9))
        plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.05, top=0.95)
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(iters, self.nsp_loss['batch'], 'g', label='nsp_loss')
        ax1.set_title('nsp loss')
        ax1.grid(True)

        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(epochs, self.nsp_loss['epoch'], 'b', label='nsp_loss')
        ax2.plot(epochs, self.val_nsp_loss['epoch'], 'g', label='val_nsp_loss')
        ax2.set_title('nsp train/valid loss')
        ax2.grid(True)
        ax2.legend(loc="upper right")

        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(iters, self.start_loss['batch'], 'g', label='start_loss')
        ax3.set_title('start loss')
        ax3.grid(True)

        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(epochs, self.start_loss['epoch'], 'b', label='start_loss')
        ax4.plot(epochs, self.val_start_loss['epoch'], 'g', label='val_start_loss')
        ax4.set_title('start train/valid loss')
        ax4.grid(True)
        ax4.legend(loc="upper right")

        ax5 = plt.subplot(4, 2, 5)
        ax5.plot(iters, self.end_loss['batch'], 'g', label='end_loss')
        ax5.set_title('end loss')
        ax3.grid(True)

        ax6 = plt.subplot(4, 2, 6)
        ax6.plot(epochs, self.end_loss['epoch'], 'b', label='end_loss')
        ax6.plot(epochs, self.val_end_loss['epoch'], 'g', label='val_end_loss')
        ax6.set_title('end train/valid loss')
        ax6.grid(True)
        ax6.legend(loc="upper right")

        ax7 = plt.subplot(4, 2, 7)
        ax7.plot(iters, self.end_loss['batch'], 'g', label='loss')
        ax7.set_title('loss')
        ax7.grid(True)

        ax7 = plt.subplot(4, 2, 8)
        ax7.plot(epochs, self.end_loss['epoch'], 'b', label='loss')
        ax7.plot(epochs, self.val_end_loss['epoch'], 'g', label='val_loss')
        ax7.set_title('train/valid loss')
        ax7.grid(True)
        ax7.legend(loc="upper right")

        plt.savefig(path)

