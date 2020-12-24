#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:54:04 2020



@author: jacobpfondevida
"""

import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

class DatasetWorker:
    
    def __init__(self, data_path, normalize=False):
        self.orig_data = pd.read_csv(data_path)
        
        self.train_set = pd.DataFrame()
        self.cv_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        
        self.Y_train = pd.DataFrame()
        self.Y_cv = pd.DataFrame()
        self.Y_test = pd.DataFrame()
        
        self.normalize = normalize
            
        self.data_segmentation()
        
    
    def collect_labels(self, input_set):        
        #collect labels for example + delete its column in the original np array
    
        labels_array = input_set['label']
        
        #updates original input_set since python passes by reference. also makes input_set mx784 instead of mx785. necessary.
        del input_set['label']
        
        return labels_array
        
    def data_segmentation(self):
        #segmenting into train/crossvalidation/test
        
        example_count = len(self.orig_data.index)
        
        train_cutoff = int(.6 * example_count)
        cv_cutoff = int(train_cutoff + 0.2 * example_count)
        #segment 60% train 20% cv 20% test
        
        self.train_set = self.orig_data[0:train_cutoff]
        self.cv_set = self.orig_data[train_cutoff:cv_cutoff]
        self.test_set = self.orig_data[cv_cutoff:]
        
        self.Y_train = to_categorical(self.collect_labels(self.train_set), num_classes=10)
        self.train_set = self.train_set.values.reshape(-1, 28, 28, 1)
        
        self.Y_cv = to_categorical(self.collect_labels(self.cv_set), num_classes=10)
        self.cv_set = self.cv_set.values.reshape(-1, 28, 28, 1)
        
        self.Y_test = to_categorical(self.collect_labels(self.test_set), num_classes=10)
        self.tesy_set = self.test_set.values.reshape(-1, 28, 28, 1)
        
        
        if self.normalize:
            self.train_set = self.normalizer(self.train_set)
            self.cv_set = self.normalizer(self.cv_set)
            self.test_set = self.normalizer(self.test_set)
        
    def normalizer(self, input_set):
    
        return input_set / 255.0
    
    #for this, input_num_classes would be 10. for 0-9
    def one_hot(self, input_set, input_num_classes):
        
        return to_categorical(input_set, num_classes = input_num_classes)