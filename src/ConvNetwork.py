#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:51:32 2020

@author: jacobpfondevida
"""

import numpy as np
import pandas as pd

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten #core layers
from keras.layers import Conv2D, MaxPool2D #CNN layers
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau

import constants

class ConvNetwork: 
    
    def __init__(self, input_dataset):
        """
        Initializes the convolution network with the inputted dataset
        

        Args:
            input_dataset (DatasetWorker): Normalized dataset that has 
            been segmented off into separate train/test/crossvalidate sets.

        Returns:
            None.

        """
        
        self.dataset = input_dataset 
        
        self.model = Sequential()
        
        self.define_model()        
        self.set_optimizer()
        
    def define_model(self):
        """
        Adds the layers and filters required for the network to function.

        Returns:
            None.

        """   
        
        #need input_shape for first layer but not additional ones
        self.model.add(Conv2D(filters = 32, kernel_size = (5, 5), 
                              padding = 'Same', activation = 'relu', 
                              input_shape = (28, 28, 1)))
        self.model.add(Conv2D(filters = 32, kernel_size = (5, 5), 
                              padding = 'Same', activation = 'relu'))
        self.model.add(MaxPool2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                              padding = 'Same', activation = 'relu'))
        self.model.add(Conv2D(filters = 64, kernel_size = (3, 3),
                              padding = 'Same', activation = 'relu'))
        self.model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation = 'softmax'))
        
    def set_optimizer(self):
        """
        Initializes the optimizer that will make the learning model
        adaptive

        Returns:
            None.

        """
        
        optimizer = RMSprop(lr = self.LR, rho = self.RHO, 
                            epsilon = self.EPSILON, decay = self.DECAY)
        
        self.model.compile(optimizer = optimizer, 
                           loss = "categorical_crossentropy", 
                           metrics = ["accuracy"])
        
        self.learning_rate_reduction = ReduceLROnPlateau(
            monitor = 'val_loss', patience = 3, verbose = 1, factor = 0.5,
            min_lr = 0.0001)
        
        self.epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        
    def data_augmentation(self):
        self.datagenerator = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                                samplewise_std_normalization=False, zca_whitening=False, rotation_range=10,
                                                zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                                horizontal_flip=False, vertical_flip=False)
        
        self.datagenerator.fit(self.dataset.train_set)
        
    def fit_cnn(self):
        """
        without using datagenerator / data augmentation
        self.model.fit(x=self.dataset.train_set, y=self.dataset.Y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data= (self.dataset.cv_set, self.dataset.Y_cv), verbose=1)
        """
        
        self.model.fit(self.datagenerator.flow(x=self.dataset.train_set, y=self.dataset.Y_train, batch_size=self.batch_size),
                       epochs=self.epochs, validation_data= (self.dataset.cv_set, self.dataset.Y_cv), verbose=1,
                       steps_per_epoch=self.dataset.train_set.shape[0] // self.batch_size, callbacks=[self.learning_rate_reduction])
                       
        