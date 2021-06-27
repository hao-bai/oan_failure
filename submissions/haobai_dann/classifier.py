#!/usr/bin/env python
# coding: utf8
''' Train the model and predict the probability of OAN weak and failure
    
    author: Hao BAI
    mail: hao.bai@insa-rouen.fr
    
    Copyright (c) 2021 Hao Bai
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Dropout



#!------------------------------------------------------------------------------
#!                                    CLASS
#!------------------------------------------------------------------------------
class Classifier:

    def __init__(self):
        ''' Fully connected deep neural network
        '''
        self.clf = DANN()
        self.batch_size = 10
        self.epochs = 5
        self.optimizer = tf.optimizers.Adam()
        self.metric = tf.keras.metrics.Precision()
        self.lamda = 1.0


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
        # self.batch_size = length #! 输入的y_target数量很少（~20）
        source_dataset = tf.data.Dataset.from_tensor_slices((X_source, y_source)).batch(self.batch_size*10, drop_remainder=True)
        da_dataset = tf.data.Dataset.from_tensor_slices((X_source[:length], y_source[:length], X_target, y_target)).batch(self.batch_size, drop_remainder=True)
        test_dataset_used = tf.data.Dataset.from_tensor_slices((X_target, y_target)).batch(self.batch_size, drop_remainder=True) #Test Dataset over Target (used for training)

        print("---- Prepare Datasets")
        print("Length of y_target:", len(y_target))
        print("source_dataset:", source_dataset,
              "length:", len(list(source_dataset.as_numpy_iterator())) )
        print("da_dataset", da_dataset,
              "length:", len(list(da_dataset.as_numpy_iterator())) )
        print("---- End ----", end="\n")
        
        # Avoid "WARNING:tensorflow:Your input ran out of data;"
        # source_dataset = source_dataset.repeat()
        # da_dataset = da_dataset.repeat()
        # Label the domain
        domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size, 1]),
                                   np.tile([0., 1.], [self.batch_size, 1])])
        domain_labels = domain_labels.astype('float32')


        ## Configure the neural network
        

        ## Train the model
        self.metric.reset_states()
        acc_list, source_acc = [], []

        print("                SOURCE ONLY")
        for epoch in range(self.epochs):
            print("\n============ EPOCH {} ============".format(epoch))
            p = float(epoch) / self.epochs
            lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
            lamda = lamda.astype('float32')
            lamda = p + 0.4
            print("lamda:", lamda)

            for batch in source_dataset:
                # print("\tbatch length:", len(batch), "batch[0]:", batch[0].shape)
                train_step_source(*batch,
                    lamda=lamda,
                    model=self.clf,
                    domain_labels=domain_labels,
                    metric=self.metric,
                    model_optimizer=self.optimizer,
                )
            
            print("Training: Epoch {} :\t Source Accuracy : {:.3%}".format(epoch, self.metric.result()), end='  |  ')
            source_acc.append(self.metric.result())
            # test(test_dataset_used)
            self.metric.reset_states()
            print("============ END EPOCH ============", end="\n")

        print("                DOMAIN ADAPTION")
        for epoch in range(self.epochs):
            print("\n============ EPOCH {} ============".format(epoch))
            p = float(epoch) / self.epochs
            lamda = 2 / (1 + np.exp(-100 * p, dtype=np.float32)) - 1
            lamda = lamda.astype('float32')
            lamda = p + 0.4
            print("lamda:", lamda)
            
            for batch in da_dataset:
                # print("\tbatch length:", len(batch), "batch[0]:", batch[0].shape)
                train_step_da(*batch,
                    lamda=lamda,
                    model=self.clf,
                    domain_labels=domain_labels,
                    metric=self.metric,
                    model_optimizer=self.optimizer,
                )
            
            print("Training: Epoch {} :\t Source Accuracy : {:.3%}".format(epoch, self.metric.result()), end='  |  ')
            acc_list.append(self.metric.result())
            # test(test_dataset_used)
            self.metric.reset_states()
            print("============ END EPOCH ============", end="\n")


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
        y_pred = self.clf(X_target, train=False, source_train=True)
        y_proba = np.hstack([1-y_pred, y_pred,])
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
        self.label_predictor_layer2 = Dense(1, activation='sigmoid')
        
        #Domain Predictor
        self.domain_predictor_layer0 = GradientReversalLayer()
        self.domain_predictor_layer1 = Dense(100, activation='relu')
        self.domain_predictor_layer2 = Dense(2, activation=None)
        
    def call(self, x, train=False, source_train=False, lamda=1.0):
        # print("x:", x)

        #* Feature Extractor
        # x = tf.keras.Input(shape=(672, 9, 1), name="Input_Layer")
        x = self.feature_extractor_layer0(x)
        # x = self.feature_extractor_layer1(x, training=train)
        x = self.feature_extractor_layer2(x)
        
        x = self.feature_extractor_layer3(x)
        # x = self.feature_extractor_layer4(x, training=train)
        # x = self.feature_extractor_layer5(x, training=train)
        x = self.feature_extractor_layer6(x)
        
        # print("x before feature:", x)
        feature = tf.reshape(x, [-1, 167 * 1 * 64])
        # print("feature:", feature)
        
        #* Label Predictor
        if source_train is True:
            feature_slice = feature
        else:
            feature_slice = tf.slice(feature, [0, 0], [feature.shape[0] // 2, -1])
        # print("feature_slice", feature_slice)
        
        lp_x = self.label_predictor_layer0(feature_slice)
        lp_x = self.label_predictor_layer1(lp_x)
        l_logits = self.label_predictor_layer2(lp_x)
        # print("l_logits", l_logits)
        
        #* Domain Predictor
        if source_train is True:
            return l_logits
        else:
            dp_x = self.domain_predictor_layer0(feature, lamda)    #GradientReversalLayer
            dp_x = self.domain_predictor_layer1(dp_x)
            d_logits = self.domain_predictor_layer2(dp_x)
            print("d_logits", d_logits)
            
            return l_logits, d_logits

def loss_func(input_logits, target_labels):
    # print("\tinput_logits:", type(input_logits), input_logits.shape)
    # print("\ttarget_labels:", type(target_labels), target_labels.shape)
    # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=input_logits, labels=target_labels))
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=input_logits, labels=target_labels))
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=input_logits, y_true=target_labels))

def get_loss(l_logits, labels, d_logits=None, domain=None):
    if d_logits is None:
        return loss_func(l_logits, labels)
    else:
        return loss_func(l_logits, labels) + loss_func(d_logits, domain)

@tf.function
def train_step_da(s_images, s_labels, t_images=None, t_labels=None, lamda=1.0,
    model=None, domain_labels=None, metric=None, model_optimizer=None):
    print("\n---- train_step_da")
    images = tf.concat([s_images, t_images], 0)
    labels = s_labels
    
    with tf.GradientTape() as tape:
        output = model(images, train=True, source_train=False, lamda=lamda)
        print("\toutput:", type(output), len(output))
        print(type(output[0]), output[0][:5])
        print(type(output[1]), output[1][:5])
        model_loss = get_loss(output[0], labels, output[1], domain_labels)
        metric(output[0], labels)
        
    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))


@tf.function
def train_step_source(s_images, s_labels, lamda=1.0,
    model=None, domain_labels=None, metric=None, model_optimizer=None):
    print("\n---- train_step_source")
    images = s_images
    labels = s_labels
    
    with tf.GradientTape() as tape:
        output = model(images, train=True, source_train=True, lamda=lamda)
        print("\toutput:", type(output), output.shape)
        model_loss = get_loss(output, labels)
        metric(output, labels)
        
    gradients_mdan = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients_mdan, model.trainable_variables))


@tf.function
def test_step(t_images, t_labels):
    print("\n---- test_step")
    images = t_images
    labels = t_labels
    
    output = model(images, train=False, source_train=True)
    epoch_accuracy(output, labels)


def test(test_dataset_used):
    epoch_accuracy.reset_states()
    
    #Target Domain (used for Training)
    for batch in test_dataset_used:
        test_step(*batch)
    
    print("[Target] Metric (used for training): {:.3%}".format(epoch_accuracy.result()))
    # test2_acc.append(epoch_accuracy.result())
    epoch_accuracy.reset_states()



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
