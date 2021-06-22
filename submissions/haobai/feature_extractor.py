#!/usr/bin/env python
# coding: utf8
''' Pre-process dataset

    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
'''
import numpy as np

class FeatureExtractor:

    def __init__(self):
        pass

    def transform(self, X):
        ''' Replace NaN by 0 and flatten the matrix to size (sample, 6720)
        
        Executed on every input data (i.e., source, bkg, target) and passed
        the resulting arrays to `fit`and `predict` methods in :class: Classifier

        :param X: dataset of (sample, time, features), i.e., (sample, 672, 10)
        :type X: ndarray
        :return: dataset of (sample, 6720)
        :rtype: ndarray
        '''
        np.nan_to_num(X, copy=False)
        return X
