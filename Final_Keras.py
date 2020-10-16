#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries, packages, etc.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[7]:


#(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
#y = y.reshape((y.shape[0], 1))
#print(X.shape)


# In[19]:


class CORblock_S:

    scale = 4  # scale of the bottleneck convolution channels-??
    chanDim = 1 #Change: Set this correctly, can also be taken as input parameter
    
    #def __init__(self, out_channels, times=1):
    def __init__(self, in_channels, out_channels, times=1):
        
        self.times = times
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    @staticmethod
    def CORblock_S_defn():

        conv_input = Conv2D(out_channels,(1,1),use_bias=False, input_shape = (,,))(inp) #Figure out input format

        for t in range(self.times):
            if t == 0:
                skip = BatchNormalization(axis=chanDim)(Conv2D(out_channels,(1,1), strides = 2, use_bias=False)(conv_input))
                conv2_stride = 2
            else:
                skip = conv_input
                conv2_stride = 1

            conv1 = Conv2D(out_channels*scale,(1,1),use_bias=False)(conv_input)
            bn1 = BatchNormalization(axis=chanDim)(conv1)
            act1 = Activation("relu")(bn1)

            conv2 = Conv2D(out_channels*scale, (3,3), strides = conv2_stride,use_bias=False)(act1)
            bn2 = BatchNormalization(axis=chanDim)(conv2)
            act2 = Activation("relu")(bn2)

            conv3 = Conv2D(out_channels,(1,1),use_bias=False)(act2)
            bn3 = BatchNormalization(axis=chanDim)(conv3)

            bn3 += skip
            act3 = Activation("relu")(bn3)
            output = act3

        return output
    
def CORnet_S():
    model = Sequential(OrderedDict([('V1', Sequential(
        [ 
            Input(shape=input_shape),
            Conv2D(64, (7,7), strides=2, padding=3,use_bias=False),
            BatchNormalization(axis=chanDim),
            Activation("relu"),
            MaxPooling2D(pool_size=(3, 3), strides=2, padding=1),
            Conv2D(64, (3,3), strides=1, padding=1,use_bias=False)
            BatchNormalization(axis=chanDim),
            Activation("relu"),
            ])),
                                    
            ('V2', CORblock_S(128, times=2)),
            ('V4', CORblock_S(256, times=4)),
            ('IT', CORblock_S(512, times=2)),
            '''('decoder', Sequential(OrderedDict([
            ('avgpool', AveragePooling2D(pool_size=(1,1))),
            ('flatten', Flatten()),
            ('linear', Dense()),
            ('output', Dense()),
             
            ]))) #Fix brackets-           
            ]))'''
         
         
#Weight initialization block below:
                                    
    


# In[ ]:


#model.summary()

#model.compile(loss = 'categorical_crossentropy', optimizer= )

#model.fit() -- add parameters

#model.save(".h5")

     ''' Structure of PyTorch code for V1
     
            Sequential(OrderedDict([  
            ('conv1', Conv2D(64, (7,7), strides=2, padding=3,use_bias=False, input_shape = ())(inp)),
            ('norm1', BatchNormalization(axis=chanDim)()),
            ('nonlin1', Activation("relu")()),
            ('pool', MaxPooling2D(pool_size=(3, 3), strides=2, padding=1)()),
            ('conv2', Conv2D(64, (3,3), strides=1, padding=1,use_bias=False)()),
            ('norm2', BatchNormalization(axis=chanDim)()),
            ('nonlin2', Activation("relu")()),
            ('output', Activation("relu")()),
        ]))),
        
        '''


# In[ ]:




