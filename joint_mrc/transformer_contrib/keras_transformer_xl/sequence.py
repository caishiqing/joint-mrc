from .backend import keras
import numpy as np

__all__ = ['MemorySequence']


class MemorySequence(keras.utils.Sequence):

    def __init__(self,
                 units: int,
                 model: keras.models.Model,
                 sequence: keras.utils.Sequence,
                 target_len: int):
        """Initialize the sequence.

        :param units: Dimension inside the model.
        :param model: The built model.
        :param sequence: The original sequence.
        :param target_len: The length of prediction.
        """
        self.units = units
        self.model = model
        self.sequence = sequence
        self.target_len = target_len

        self.indice = []
        for i in range(len(sequence)):
            item = sequence[i]
            length = self._get_first_shape(item)[1]
            number = (length + target_len - 1) // target_len
            for j in range(number):
                self.indice.append((i, j))

        self.last_index, self.last_item = -1, None

        self.memory_length_index = None
        for i, input_layer in enumerate(model.inputs):
            name = input_layer.name.split(':')[0].split('/')[0]
            if name.startswith('Input-Memory-Length'):
                self.memory_length_index = i
                break

    def __len__(self):
        return len(self.indice)

    def __getitem__(self, index):
        sub_index, sub_num = self.indice[index]
        if sub_index == self.last_index:
            item = self.last_item
        else:
            item = self.sequence[sub_index]
            self.last_index = sub_index
            self.last_item = item
        start = sub_num * self.target_len
        stop = start + self.target_len
        s = slice(start, stop)
        batch_size = self._get_first_shape(item)[0]

        if isinstance(item[0], (list, tuple)):
            inputs = [self._pad_target(sub_item[:, s, ...]) for sub_item in item[0]]
        else:
            inputs = [self._pad_target(item[0][:, s, ...])]
        memory_length_input = np.ones(batch_size) * sub_num * self.target_len
        inputs = inputs[:self.memory_length_index] + [memory_length_input] + inputs[self.memory_length_index:]

        if isinstance(item[1], (list, tuple)):
            outputs = [self._pad_target(sub_item[:, s, ...]) for sub_item in item[1]]
        else:
            outputs = self._pad_target(item[1][:, s, ...])
        return inputs, outputs

    @staticmethod
    def _get_first_shape(item):
        if isinstance(item[0], (list, tuple)):
            return item[0][0].shape
        return item[0].shape

    def _pad_target(self, item: np.ndarray):
        length = item.shape[1]
        if length != self.target_len:
            if item.ndim == 2:
                return np.pad(item, ((0, 0), (0, self.target_len - length)), 'constant', constant_values=0)
            return np.pad(item, ((0, 0), (0, self.target_len - length), (0, 0)), 'constant', constant_values=0)
        return item
