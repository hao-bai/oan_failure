#!/usr/bin/env python
# coding: utf8
''' Pre-process dataset

    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
import numpy as np


#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class FeatureExtractor:

    def __init__(self):
        pass

    def transform(self, X):
        ''' Replace NaN by 0 and flatten the matrix to size (sample, 6720).
        Executed on every input data (i.e., source, bkg, target) and passed
        the resulting arrays to `fit`and `predict` methods in :class: Classifier

        Parameters
        ----------
        `X`: ndarray of (sample, 672, 10)
            3D input dataset(sample, time, features)
        
        Returns
        -------
        `X`: ndarray of (sample, 6720)
            The filtered dataset
        '''
        np.nan_to_num(X, copy=False)
        X = X.reshape(X.shape[0], -1)
        return X



#!------------------------------------------------------------------------------
#!                                     FUNCTION
#!------------------------------------------------------------------------------
def main():
    pass



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
