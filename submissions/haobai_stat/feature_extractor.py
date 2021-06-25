#!/usr/bin/env python
# coding: utf8
''' Pre-process dataset

    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
import numpy as np
from scipy import stats



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class FeatureExtractor:

    def __init__(self):
        pass

    def transform(self, X):
        ''' Compute the statstics of each feature and flatten the matrix to size
        (sample, 6720).
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
        X = X.astype(np.float64)
        tmp_X = []
        for x in X:
            # Deal NaN
            x[:, np.all(~np.isfinite(x), axis=0)] = 0 # Columns with all NaN
            # Compute statistics
            _ = []
            if np.any(~np.isfinite(x)) == True:
                b = []
                for row in x.T:
                    tmp = row[np.isfinite(row)]
                    pct = np.percentile(tmp, [25, 50, 75])
                    b.append(pct)
                b = np.array(b).T
            else:
                b = np.percentile(x, [25, 50, 75], axis=0)
            _.append( b[0] ) # percentile@25%
            _.append( b[1] ) # percentile@50%
            _.append( b[2] ) # percentile@75%
            _.append( np.nanmean(x, axis=0) ) # mean
            _.append( np.nanstd(x, axis=0)) # standard deviation
            _.append( np.nanmax(x, axis=0) ) # maximum
            _.append( np.nanmin(x, axis=0) ) # minnimum
            _.append( stats.mode(x, axis=0, nan_policy="omit")[0][0]) # mode
            _.append( stats.kurtosis(x, axis=0, nan_policy="omit", fisher=False)) # kurtosis
            _.append( stats.skew(x, axis=0, nan_policy="omit")) # skewness
            _.append( np.sum(np.isfinite(x), axis=0)/x.shape[0] ) # ratio of finite values
            # Aggregate data
            tmp_X.append( np.array(_) )
        X = np.array(tmp_X)
        # Flatten
        X = X.reshape(X.shape[0], -1)
        # print("Should be True: ", np.all(np.isfinite(X)))
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
