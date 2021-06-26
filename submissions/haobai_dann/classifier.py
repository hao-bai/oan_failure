#!/usr/bin/env python
# coding: utf8
''' Train the model and predict the probability of OAN weak and failure
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class Classifier:

    def __init__(self):
        ''' Fully connected deep neural network
        '''
        model = DANN()
        self.clf = model
        self.batch_size = 32


    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        ''' Train the model

        Parameters
        ----------
        `X_source`: ndarray of (sample, 672, 9, 1)
            labeled dataset (weak or failure) from source (city A)
        `X_source_bkg`: ndarray of (sample, 672, 9, 1)
            dataset (good class) from source (city A)
        `X_target`: ndarray of (sample, 672, 9, 1)
            labeled dataset (weak or failure) from target (city B)
        `X_target_bkg`: ndarray of (sample, 672, 9, 1)
            dataset (good class) from target (city B)
        `y_source`: ndarray of (sample, )
            labels from source (city A)
        `y_target`: ndarray of (sample, )
            labels from targete (city B)
        '''
        ## Prepare dataset
        # Reshape labels to (sample, 1)
        y_source = y_source.reshape(-1, 1)
        y_target = y_target.reshape(-1, 1)
        # Generate TensorFlows dataset
        length = y_target.shape[0]
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, y_source)).shuffle(100).batch(self.batch_size, drop_remainder=True)
        da_dataset = tf.data.Dataset.from_tensor_slices((X_source[:length], y_source[:length], X_target, y_target)).shuffle(100).batch(self.batch_size, drop_remainder=True)
        test_dataset2 = tf.data.Dataset.from_tensor_slices((X_target, y_target)).shuffle(100).batch(self.batch_size, drop_remainder=True) #Test Dataset over Target (used for training)
        # Avoid "WARNING:tensorflow:Your input ran out of data;"
        source_dataset = source_dataset.repeat()
        da_dataset = da_dataset.repeat()
        test_dataset2 = test_dataset2.repeat()

        ## Configure the neural network
        # TODO
        self.clf.fit(train_dataset, epochs=100, steps_per_epoch=200,
            validation_data=valid_dataset, validation_steps=3, )


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
        y_pred = self.clf.predict(X_target)
        y_proba = np.hstack([1-y_pred, y_pred])
        return y_proba



#!------------------------------------------------------------------------------
#!                        DOMAIN ADVERSARIAL NEURAL NETWORK
#!------------------------------------------------------------------------------
#Gradient Reversal Layer
@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)
    
    def grad(dy):
        return lamda * -dy, None
    
    return y, grad


class GradientReversalLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
    
    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)


class DANN(Model):

    def __init__(self):
        super().__init__()
        
        #Feature Extractor
        self.feature_extractor_layer0 = Conv2D(32, 2, activation='relu')
        self.feature_extractor_layer1 = BatchNormalization()
        self.feature_extractor_layer2 = MaxPool2D(pool_size=(2, 2),)
        
        self.feature_extractor_layer3 = Conv2D(64, 2, activation='relu')
        self.feature_extractor_layer4 = Dropout(0.5)
        self.feature_extractor_layer5 = BatchNormalization()
        self.feature_extractor_layer6 = MaxPool2D(pool_size=(2, 2),)
        
        #Label Predictor
        self.label_predictor_layer0 = Dense(100, activation='relu')
        self.label_predictor_layer1 = Dense(100, activation='relu')
        self.label_predictor_layer2 = Dense(1, activation=None) # logits shape (1336, 10)
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = Dense(100, activation='relu')
        self.domain_predictor_layer2 = Dense(2, activation=None)
        
    def call(self, x, train=False, source_train=False, lamda=1.0):
        print("x:", x)

        #Feature Extractor
        # x = tf.keras.Input(shape=(672, 9, 1), name="Input_Layer")
        x = self.feature_extractor_layer0(x)
        x = self.feature_extractor_layer1(x, training=train)
        x = self.feature_extractor_layer2(x)
        
        x = self.feature_extractor_layer3(x)
        x = self.feature_extractor_layer4(x, training=train)
        x = self.feature_extractor_layer5(x, training=train)
        x = self.feature_extractor_layer6(x)
        
        print("x before feature:", x)
        feature = tf.reshape(x, [-1, 167 * 1 * 64])
        print("feature:", feature)
        
        #Label Predictor
        if source_train is True:
            feature_slice = feature
        else:
            feature_slice = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])
        print("feature_slice", feature_slice)
        
        lp_x = self.label_predictor_layer0(feature_slice)
        lp_x = self.label_predictor_layer1(lp_x)
        l_logits = self.label_predictor_layer2(lp_x)
        print("l_logits", l_logits)
        
        #Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dp_x = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_logits = self.domain_predictor_layer2(dp_x)
            print("d_logits", d_logits)
            
            return l_logits, d_logits





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
