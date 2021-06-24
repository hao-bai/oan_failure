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
            # 处理NaN数据
            x[:, np.all(~np.isfinite(x), axis=0)] = 0 # 用0填充全部是NaN的列
            # ------------------------------------------------------------------
            # 计算统计量
            _ = []
            # There is a bug in `np.nanpercentile` which computes very slow
            if np.any(~np.isfinite(x)) == True:
                b = []
                for row in x.T:
                    tmp = row[np.isfinite(row)]
                    pct = np.percentile(tmp, [25, 50, 75])
                    b.append(pct)
                b = np.array(b).T
            else:
                b = np.percentile(x, [25, 50, 75], axis=0)
            _.append( b[0] ) # 一分位数@25
            _.append( b[1] ) # 二分位数
            _.append( b[2] ) # 三分位数@75

            # with warnings.catch_warnings():
            #     warnings.filterwarnings('error')
            #     try:
            _.append( np.nanmean(x, axis=0) ) # 均值
            _.append( np.nanstd(x, axis=0)) # 标准差
            _.append( np.nanmax(x, axis=0) ) # 最大值
            _.append( np.nanmin(x, axis=0) ) # 最小值
            _.append( stats.mode(x, axis=0, nan_policy="omit")[0][0]) # 众数
            _.append( stats.kurtosis(x, axis=0, nan_policy="omit", fisher=False)) # 峰度 # RuntimeWarning: overflow => change type to np.float64
            _.append( stats.skew(x, axis=0, nan_policy="omit")) # 偏度
            _.append( np.sum(np.isfinite(x), axis=0)/x.shape[0] ) # 有效值数量占比
                # except Warning as e:
                #     print("x is", x.shape)
                #     print("_ is", len(_))
                #     TEST = x
                #     raise e
            # ------------------------------------------------------------------
            # 加入第3维数组
            tmp_X.append( np.array(_) )
            # tmp_X.append( x )
        X = np.array(tmp_X)

        # flatten
        X = X.reshape(X.shape[0], -1) # required for outlier detection
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
