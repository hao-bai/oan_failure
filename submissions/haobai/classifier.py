#!/usr/bin/env python
# coding: utf8
''' Train the model and predict values
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
'''
from sklearn.ensemble import RandomForestClassifier



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class Classifier:

    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=2, max_depth=2, random_state=44, n_jobs=-1)

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        ''' Random forest model

        :param X_source: dataset of (sample, 6720)
        :type X_source: ndarray
        '''
        self.clf.fit(X_source, y_source)

    def predict_proba(self, X_target, X_target_bkg):
        ''' Use the trained model to predict on target dataset

        :param X_target: dataset of (sample, 6720)
        :type X_target: ndarray
        '''
        y_proba = self.clf.predict_proba(X_target)
        return y_proba



#!------------------------------------------------------------------------------
#!                                     FUNCTION
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------



# %%
