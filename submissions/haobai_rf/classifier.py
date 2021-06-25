#!/usr/bin/env python
# coding: utf8
''' Train the model and predict the probability of OAN weak and failure
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
from sklearn.ensemble import RandomForestClassifier
import imblearn as il



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class Classifier:

    def __init__(self, **kwargs):
        ''' Random Forest
        '''
        model = RandomForestClassifier(
            n_estimators=2, max_depth=2, n_jobs=-1)
        self.clf = model
        self.sampling_strategy_over = kwargs.get("sampling_strategy_over", None)
        self.sampling_strategy_under = kwargs.get("sampling_strategy_under", None)

    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        ''' Train the model

        Parameters
        ----------
        `X_source`: ndarray of (sample, 6720)
            labeled dataset (weak or failure) from source (city A)
        `X_source_bkg`: ndarray of (sample, 6720)
            dataset (good class) from source (city A)
        `X_target`: ndarray of (sample, 6720)
            labeled dataset (weak or failure) from target (city B)
        `X_target_bkg`: ndarray of (sample, 6720)
            dataset (good class) from target (city B)
        `y_source`: ndarray of (sample, )
            labels from source (city A)
        `y_target`: ndarray of (sample, )
            labels from targete (city B)
        '''
        # Oversampling & undersampling
        over = il.over_sampling.RandomOverSampler(sampling_strategy=self.sampling_strategy_over)
        X_source, y_source = over.fit_resample(X_source, y_source)
        
        under = il.under_sampling.RandomUnderSampler(sampling_strategy=self.sampling_strategy_under) # 0.2 or 0.3
        # under = il.under_sampling.AllKNN(sampling_strategy=self.sampling_strategy) # ValueError: 'clean-sampling' methods do let the user specify the sampling ratio
        # under = il.under_sampling.TomekLinks(sampling_strategy=self.sampling_strategy) # long time
        X_source, y_source = under.fit_resample(X_source, y_source)
        
        # Train the model
        self.clf.fit(X_source, y_source)


    def predict_proba(self, X_target, X_target_bkg):
        ''' Use the trained model to predict on target dataset

        Parameters
        ----------
        `X_target`: ndarray of (sample, 6720)
            labeled dataset (weak or failure) from target (city B)
        `X_target_bkg`: ndarray of (sample, 6720)
            dataset (good class) from target (city B)
        
        Returns
        -------
        `y_proba`: ndarray of (sample, 2)
            The class probabilities of the input samples. The order of the 
            classes corresponds to [0 (weak), 1 (failure)]
        '''
        y_proba = self.clf.predict_proba(X_target)
        return y_proba



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
