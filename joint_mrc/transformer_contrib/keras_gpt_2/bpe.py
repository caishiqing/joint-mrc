# coding=utf8
from __future__ import unicode_literals
import json
import codecs
import regex as re


__all__ = ['BytePairEncoding', 'get_bpe_from_files']


try:
    chr = unichr
except Exception as e:
    '''No need to use `unichr` in Python 3'''


class BytePairEncoding(object):

    def __init__(self,
                 token_dict,
                 bpe_rank):
        """Encode and decode of BPE.

        :param token_dict: Maps from encoded token to indices.
        :param bpe_rank: Maps from byte pair to an integer rank.
        """
        self.token_dict = token_dict
        self.token_dict_inv = {v: k for k, v in self.token_dict.items()}
        self.bpe_rank = bpe_rank
        self.byte_encoder = self.init_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.token_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        self.cache = {}

    @staticmethod
    def init_byte_encoder():
        codes = list(range(ord("!"), ord("~") + 1)) +\
                list(range(ord("¡"), ord("¬") + 1)) +\
                list(range(ord("®"), ord("ÿ") + 1))
        byte_encoder = {code: chr(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr(2 ** 8 + shift)
                shift += 1
        return byte_encoder

    def get_bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self.bpe_rank.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self.bpe_rank:
                break
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self.cache[token] = chars
        return chars

    def encode(self, text):
        indices = []
        for token in re.findall(self.token_pattern, text):
            token = bytearray(token.encode('utf-8'))
            chars = ''.join(self.byte_encoder[code] for code in token)
            indices += [self.token_dict[token] for token in self.get_bpe(chars)]
        return indices

    def decode(self, tokens):
        text = ''.join([self.token_dict_inv[token] for token in tokens])
        return bytearray([self.byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')


def get_bpe_from_files(encoder_path, vocab_path):
    """Get initialized BPE.

    :param encoder_path: Path to 'encoder.json'.
    :param vocab_path: Path to 'vocab.bpe'
    :return: The object from encode and decode strings.
    """
    with codecs.open(encoder_path, 'r', 'utf8') as reader:
        token_dict = json.load(reader)
    bpe_rank = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        reader.readline()
        for rank, line in enumerate(reader):
            line = line.strip()
            if line:
                bpe_rank[tuple(line.split())] = rank
    return BytePairEncoding(token_dict, bpe_rank)
