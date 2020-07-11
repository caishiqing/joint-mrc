# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import
from math import exp, log
from copy import copy
from collections import Counter
import re, os
import numpy as np

__all__ = [
    'lcs', 'levenshtein', 'F1_score', 'exact_match',
    'Bleu_4', 'Rouge_L', 'evaluate',
]

EPSILON = 1e-7

def lcs(s1, s2):
    """ 最长公共子序列 """
    v1 = [0] * (len(s2) + 1)
    v2 = [0] * (len(s2) + 1)
    for i in range(len(s1)):
        v1 = v2[:]
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                v2[j + 1] = v1[j] + 1
            else:
                v2[j + 1] = max(
                    v1[j + 1],
                    v2[j],
                    v1[j])
    return v2[-1]

def levenshtein(s1, s2):
    """ 编辑距离 """
    v1 = list(range(len(s2) + 1))
    v2 = list(range(len(s2) + 1))
    for i in range(len(s1)):
        v1 = v2[:]
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                v2[j + 1] = v1[j]
            else:
                v2[j + 1] = min(
                    v1[j + 1] + 1,  # 插入
                    v2[j] + 1,  # 删除
                    v1[j] + 1, # 替换
                    )
    return v2[-1]


def F1_score(prediction, ground_truth):
    """
    Args:
        prediction: predict tokens/text
        ground_truth: reference tokens/text
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / (EPSILON + len(prediction))
    recall = 1.0 * num_same / (EPSILON + len(ground_truth))
    f1 = (2 * precision * recall) / (precision + recall + EPSILON)
    return f1

def exact_match(prediction, ground_truth):
    return prediction == ground_truth

def p_ngram(prediction, ground_truth, n):
    len_p = len(prediction)
    len_g = len(ground_truth)
    prediction = [prediction[i:i + n] for i in range(len_p - n)]
    ground_truth = [ground_truth[i:i + n] for i in range(len_g - n)]
    p = 0.0
    len_p = len(prediction)
    len_g = len(ground_truth)
    for pred in prediction:
        if pred in ground_truth:
            p += 1
            ground_truth.remove(pred)
    return p / (max(len_p, len_g) + EPSILON)

def Bleu_4(prediction, ground_truth):
    """ bleu_4 score """
    p = 1.0 / 4 * sum([log(p_ngram(prediction, ground_truth, n) \
                                + EPSILON) for n in range(1,5)])
    return exp(p)

def Rouge_L(prediction, ground_truth, beta=1.2):
    """ rouge_L score """
    _lcs = lcs(prediction, ground_truth)
    len_g = len(ground_truth)
    len_p = len(prediction)
    R_lcs = _lcs / float(len_g) if len_g > 0 else 0.
    P_lcs = _lcs / float(len_p) if len_p > 0 else 0.
    F_lcs = (1 + beta ** 2) * R_lcs * P_lcs / (R_lcs + (beta ** 2) * P_lcs + EPSILON)
    return F_lcs

def evaluate(match_fn, predictions, ground_truths):
    """ 评估函数

    Args:
        match_fn (function): 基础评估函数
        predictions (list): 预测样本
        ground_truths (list): 目标样本

    Returns:
        float: 评估值
    """
    res = 0.0
    for i in range(len(predictions)):
        res += match_fn(predictions[i], ground_truths[i])
    return res / (len(predictions) + EPSILON)


if __name__ == '__main__':
    s1 = 'abcdefgh'
    s2 = 'acdeffhs'
    print(lcs(s1, s2))
    print(levenshtein(s1, s2))
