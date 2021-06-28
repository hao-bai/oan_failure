#!/usr/bin/env python
# coding: utf8
''' Pre-process dataset

    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
import numpy as np
from numpy import newaxis
import warnings
warnings.filterwarnings("ignore")



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class FeatureExtractor:

    def __init__(self):
        pass

    def transform(self, X):
        ''' Deal with NaN and flatten the matrix to size (sample, 6720).
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
        #! ATTENTION
        # The idea is supposed to eliminate the common columns filled entirely 
        # by NaN. But in this competition, since we don't have access to
        # `OpticalDataset` object, it's impossible to communicate informations
        # between datasets. So, here it deletes columns that are found on public
        # dataset.
        X = np.delete(X, [3,], axis=2)
        X = X.astype(np.float64)
        
        ## 1st round
        X1, nanmean = [], []
        for i in range(X.shape[0]):
            x = X[i]
            indice = ~np.isfinite(x)
            nanmean.append(np.nanmean(x, axis=0))

            # Columns with full Nan
            col_is_nan = np.all(indice, axis=0)
            if (col_is_nan == True).any():
                X1.append(x) # deal later
                continue
            
            # Rows with full Nan
            # Unachievable. Cause we don't have access to manipulate on labels
            # row_is_nan = np.all(indice, axis=1)
            # if (row_is_nan == True).any():
            #     row = np.where(row_is_nan == True)[0]
            #     if len(row) >= x.shape[0]/4: # drop sample, /2=85%+, /4=75%+
            #         continue
            
            # Columns with partial NaN
            part_is_nan = np.any(indice, axis=0)
            if (part_is_nan == True).any():
                col = np.where(part_is_nan == True)[0]
                # part_nan[i] = col[0]
                for c in col:
                    this = x[:,c]
                    finite = this[np.isfinite(this)]
                    fill = np.repeat(finite, np.ceil(len(this)/len(finite)))[:len(this)]
                    x[:,c] = np.where(np.isfinite(this), this, fill)
            
            # Construct new array
            X1.append(x)
        X1, nanmean = np.array(X1), np.array(nanmean)

        ## 2nd round
        candidate_mean = []
        for i in range(nanmean.shape[1]):
            col = nanmean[i]
            finite = col[np.isfinite(col)]
            candidate_mean.append(finite)

        X2 = []
        for i in range(X1.shape[0]):
            x = X[i]
            indice = ~np.isfinite(x)
            # Columns with full Nan
            col_is_nan = np.all(indice, axis=0)
            if (col_is_nan == True).any():
                col = np.where(col_is_nan == True)[0]
                for c in col:
                    value = np.random.choice(candidate_mean[c])
                    x = np.nan_to_num(x, nan=value)
            X2.append(x)
        
        X = np.array(X2)

        ## Final
        X = X.reshape(X.shape[0], -1) # Flatten
        # print("Expected True:", np.all(np.isfinite(X))) # expected True
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
