#!/usr/bin/env python
# coding: utf8
''' Train the model and predict the probability of OAN weak or failure
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
'''
import tensorflow as tf
from tensorflow.keras import layers



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class Classifier:

    def __init__(self):
        model = tf.keras.Sequential()
        model.add( layers.Flatten(input_shape=(672, 10,), name="Input_Layer") )

        num_fully_connected_layers = 10
        for i in range(num_fully_connected_layers):
            model.add( layers.Dense(256, activation="relu", name="Layer{}".format(i+1)) )

        model.add( layers.Dropout(0.5, name="Layer-1"))
        model.add( layers.Dense(1, activation='softmax', name="Output_Layer") )
        model.compile(optimizer="Adam",
            loss='binary_crossentropy',
            metrics=['acc'])
        self.clf = model


    def fit(self, X_source, X_source_bkg, X_target, X_target_unlabeled,
            X_target_bkg, y_source, y_target):
        ''' Fully connected deep neural network

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
        # 将数据集转换为TensorFlow格式
        train_dataset = tf.data.Dataset.from_tensor_slices((X_source, y_source)).batch(16)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_target, y_target)).batch(16)
        # 额外操作
        train_dataset = train_dataset.map( lambda x, y: (tf.image.random_flip_left_right(x), y) )
        train_dataset = train_dataset.repeat()
        valid_dataset = valid_dataset.repeat()
        # 训练模型
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
        y_proba = self.clf.predict(X_target)
        return y_proba



#!------------------------------------------------------------------------------
#!                                     FUNCTION
#!------------------------------------------------------------------------------



#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------


