#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries, packages, etc.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[ ]:


#Create 'toy' dataset to check whether model compiles

X = np.random.randint(0, 1, size=(50,224,224,3))
Y = np.random.choice([0, 1], size=(50,))
#print(X.shape)
#print(Y.shape)

(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25, random_state=42)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

#Make_blobs
#(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
#y = y.reshape((y.shape[0], 1))
#print(X.shape)
#print(y.shape)


# In[ ]:


#Block used for V2, V4 and IT

class CORblock_S:

    scale = 4  # Scale of the bottleneck convolution channels
    chanDim = -1 #If format is 'channels last'
    
    def __init__(self, out_channels, inputs, times=1):
        #super().__init__()
        
        
        self.out_channels = out_channels   
        self.inputs = inputs
        self.times = times
        
        self.conv_input = tf.keras.layers.Conv2D(self.out_channels,(1,1)) 
        self.skip = tf.keras.layers.Conv2D(self.out_channels,(1,1), strides = 2, use_bias=False)
        self.conv1 = tf.keras.layers.Conv2D(self.out_channels*self.scale,(1,1),use_bias=False)
        #self.act1 = tf.keras.layers.Activation("relu")
        self.conv2 = tf.keras.layers.Conv2D(self.out_channels*self.scale, (3,3), strides = 2,use_bias=False)
        #self.act2 = tf.keras.layers.Activation("relu")
        self.conv3 = tf.keras.layers.Conv2D(self.out_channels,(1,1),use_bias=False)
        #self.act3 = tf.keras.layers.Activation("relu")
             
    #@staticmethod 
    def CORblock_forward(self):
        
        x = self.conv_input(self.inputs) 

        for t in range(self.times):
            
            if t == 0:
                self.skip = tf.keras.layers.BatchNormalization(axis=-1)(self.skip(x))
                conv2_stride = 2
            else:
                self.skip = x
                conv2_stride = 1
                
            x = self.conv1(x)
            x = tf.keras.layers.BatchNormalization(axis=-1)(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = self.conv2(x)
            x = tf.keras.layers.BatchNormalization(axis=-1)(x)
            x = tf.keras.layers.Activation("relu")(x)

            x = self.conv3(x)
            x = tf.keras.layers.BatchNormalization(axis=-1)(x)

            x += self.skip #Check tf.concat() also
            x = tf.keras.layers.Activation("relu")(x)
            #output = x

        return x


# In[ ]:


#Define & implement model 

#Functional API format
inputs = tf.keras.Input(shape = (224,224,3))
x = tf.keras.layers.Conv2D(64, (7,7), strides=2, padding='valid',use_bias=False)(inputs)#Check
x = tf.keras.layers.BatchNormalization(axis=-1)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same',use_bias=False)(x)
x = tf.keras.layers.BatchNormalization(axis=-1)(x)
x = tf.keras.layers.Activation("relu")(x)
    
#V2 
inputs = x
x = CORblock_S(128, inputs, times =2).CORblock_forward()

#V4
inputs = x
x = CORblock_S(256, inputs, times =4).CORblock_forward()

#IT
inputs = x
x = CORblock_S(512, inputs, times =2).CORblock_forward()
    
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1000)(x)
    
model = tf.keras.Model(inputs,x,name = 'CORnetS')

opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) #Add correct learning rate
model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])
    
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=2, batch_size=10) #Paper- 43 epochs training
score = model.evaluate(testX, testY, verbose=0)

model.summary()


# In[ ]:




