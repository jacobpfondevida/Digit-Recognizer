#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jacobpfondevida
"""

#arguments to pass into RMSprop to initialize optimizer
LR = 0.01
RHO = 0.9
EPSILON = 1e-08
DECAY = 0.0  

#increasing this value will allow the model to run more, improving accuracy
NUM_EPOCHS = 1

#number of datapoints to be processed before weight change is made
BATCH_SIZE = 86