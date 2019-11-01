# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import
from math import exp, log
from copy import copy
from collections import Counter
import re, os
import numpy as np
from ctypes import *

__all__ = [
    'f2h', 'LCS', 'F1_score', 'exact_match', 'Bleu_4', 'Rouge_L', 'evaluate',
]

EPSILON = 1e-7

def abs_path(name):
    return os.path.normpath(os.path.join(os.getcwd(),
                                         os.path.dirname(__file__),
                                         name)
                            )

strProcess = cdll.LoadLibrary(abs_path('strProcess.so'))
f2h = strProcess.fullToHalf
f2h.restype = c_wchar_p

LCS = strProcess.LCS
LCS.restype = c_int

edit_distance = strProcess.editDistance
edit_distance.restype = c_int

def match(s1, s2):  #match two string by normed of the shorter length
    return LCS(s1, s2) / float(min(len(s1), len(s2)) + EPSILON)

#prediction: predict tokens/text
#ground_truth: reference tokens/text
def F1_score(prediction, ground_truth):
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
    prediction = [prediction[i:i+n] for i in range(len_p-n)]
    ground_truth = [ground_truth[i:i+n] for i in range(len_g-n)]
    p = 0.0
    len_p = len(prediction)
    len_g = len(ground_truth)
    for pred in prediction:
        if pred in ground_truth:
            p += 1
            ground_truth.remove(pred)
    return p / (max(len_p, len_g) + EPSILON)

def Bleu_4(prediction, ground_truth):  #bleu_4 score
    c = len(prediction)
    r = len(ground_truth)
    p = 1.0 / 4 * sum([log(p_ngram(prediction, ground_truth, n) \
                                + EPSILON) for n in range(1,5)])
    return exp(p)

def Rouge_L(prediction, ground_truth, beta=1.2):  #ROUGE_L score
    lcs = LCS(prediction, ground_truth)
    len_g = len(ground_truth)
    len_p = len(prediction)
    R_lcs = lcs / float(len_g) if len_g > 0 else 0.
    P_lcs = lcs / float(len_p) if len_p > 0 else 0.
    F_lcs = (1 + beta ** 2) * R_lcs * P_lcs / (R_lcs + (beta ** 2) * P_lcs + EPSILON)
    return F_lcs

#process batch of samples by function 'match_fn'
def evaluate(match_fn, predictions, ground_truths): 
    res = 0.0
    for i in range(len(predictions)):
        res += match_fn(predictions[i], ground_truths[i])
    return res / (len(predictions) + EPSILON)


