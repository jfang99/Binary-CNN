#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential,regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
tf.config.run_functions_eagerly(True)
import numpy as np
from sklearn import linear_model 
from tensorflow.keras import initializers



def get_binary_filter(W,M):
    '''
    To get binary filter
    Input:
        W：original weight matrix W，shape=[w,h,cin,cout]
        M: number of binary filters
    Output:
        B: binarized matrices B,shape=[w,h,cin,cout]*num_of_binary_filters
    '''
    mean,var= tf.nn.moments(W, axes=list(range(len(W.get_shape()))))
    std = tf.sqrt(var)
    B=[]
    for i in range(M):
        ui = -1 + i*2/(M-1)
        B.append(tf.sign((W-mean)+ui*std))
    B=tf.convert_to_tensor(B)
    return B,mean,std

# In[ ]:


def optimize_alpha(W,B,M):
    '''
    To get optimal alpha by minimizing mse loss(linear regression)
    Input: 
        M:num_of_binary_filters
        W：original weight matrix W，shape=[w,h,cin,cout]
        B: binarized matrices B,shape=[w,h,cin,cout]*num_of_binary_filters
    Output:
        alpha,shape=[M,]
    '''
    B_reshape=tf.reshape(B,[-1,M])
    W_reshape=tf.reshape(W,[-1,1])
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(B_reshape, W_reshape)
    alpha=model.coef_[0]
    alpha=tf.convert_to_tensor(alpha)
    return alpha,B_reshape,W_reshape


# In[ ]:


def ApproxConv(binary_filter,A,alpha,bias,M,stride=1,padding="SAME"):
    '''
    To calculate weighted sum of convolution(A,Bi)
    Input:
        binray_filter: shape=[w,h,cin,cout]*M
        alpha: [M]
        A: same as X.shape
        bias: [1]
    Output:
        Shape:depending on stride and padding
    '''
    if bias==None:
        b=0
    else:
        b=bias
    binary_filter=tf.cast(binary_filter,dtype=tf.float32)
    A=tf.cast(A,dtype=tf.float32)
    O=alpha[0]*(tf.nn.conv2d(A,binary_filter[0],strides=[1, stride, stride, 1],padding=padding)+b)
    for i in range(1,M):
        O+=alpha[i]*(tf.nn.conv2d(A,binary_filter[i],strides=[1, stride, stride, 1],padding=padding)+b)
    return O


# In[ ]:


def activation_and_conv(X,beta,v,M,N,B,alpha,bias,stride):
    '''
    Input 
        beta: (N,)
        v: (N,)
        alpha: (M,)
        M: (1,)
        N: (1,)
        B: (5,7,7,32,64) 5 binary filters
        bias: (N,)
    
    Output Shape: Same as ApproxConv
    '''
    A0=tf.sign(tf.clip_by_value(X+v[0],0.,1.)-0.5)
    ConvWR = tf.zeros_like(ApproxConv(B,A0, alpha,bias[0],M,stride))
    for n in range(N):
        An = tf.sign(tf.clip_by_value(X+v[n],0.,1.)-0.5)
        ConvWR = tf.add(ConvWR, tf.multiply(ApproxConv(B,An,alpha,bias[n],M,stride),beta[n]))
    return ConvWR 


class ABCLayer(tf.keras.layers.Layer):
    def __init__(self, M, N, B1, bias1, W1,stride):
        tf.config.run_functions_eagerly(True)
        super(ABCLayer, self).__init__()
        self.M = M
        self.N = N
        #self.B_initializer=initializers.Constant(B1)
        self.B = B1
        self.bias_initializer=initializers.Constant(bias1)
        self.W_initializer=initializers.Constant(W1)
        self.alpha = tf.Variable(tf.random.normal(shape=(M,1)))
        self.Bshape=list(B1.shape)
        self.biasshape=list(bias1.shape)
        self.Wshape=list(W1.shape)
        self.stride=stride
    def build(self, input_shape):
        self.beta = self.add_weight("beta",
                                    shape = [self.N])
        self.v = self.add_weight("v",
                                shape = [self.N])
        
#         self.B = self.add_weight("B",
#                                 shape = self.Bshape,
#                                 initializer = self.B_initializer)
        
        self.bias = self.add_weight("bias",
                                   shape = self.biasshape,
                                   initializer = self.bias_initializer)
        self.W = self.add_weight("W",
                                shape = self.Wshape,
                                initializer = self.W_initializer)
        super(ABCLayer, self).build(input_shape)
    def call(self, inputs):
        Z= get_binary_filter(self.W,self.M)
        self.B=Z[0]
        Y=optimize_alpha(self.W,self.B,self.M)
        self.alpha=Y[0]
        return activation_and_conv(inputs,self.beta,self.v,self.M,self.N,self.B,self.alpha,self.bias,self.stride)







class CellBlock(layers.Layer):
    def __init__(self, filter_num, M, N, B1, bias1, W1,stride=1,abc=True):
        super(CellBlock, self).__init__()
        tf.config.run_functions_eagerly(True)
        if abc==True:
            self.conv1 = ABCLayer(M, N, B1[0], bias1[0], W1[0],stride)
        else:
            self.conv1 = Conv2D(filter_num, (3,3), strides=stride,            padding='same',kernel_initializer=keras.initializers.Constant(W1[0]),bias_initializer=keras.initializers.Constant(bias1[0]))
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        #precise
        if abc==True:
            self.conv2 = ABCLayer(M, N, B1[1], bias1[1], W1[1],stride=1)
        else:
            self.conv2 = Conv2D(filter_num, (3,3), strides=1,            padding='same',kernel_initializer=keras.initializers.Constant(W1[1]),bias_initializer=keras.initializers.Constant(bias1[1]))
        #Conv2D(filter_num, (3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        if stride !=1:
            self.residual = Conv2D(filter_num, (1,1),strides=stride)
        else:
            self.residual = lambda x:x
        
    def call (self, inputs, training=None):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output




class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes,M, N, B, bias, W):
        super(ResNet, self).__init__()

        self.stem =ABCLayer(M, N, B[0], bias[0], W[0],stride=2)
        self.stem2= Sequential([
            MaxPooling2D((3,3), strides=(2,2), padding='same'),
            BatchNormalization(),
        ]) 

        self.layer1 = self.build_cellblock(64, layers_dims[0],M, N, B[1:5], bias[1:5], W[1:5],abc=True) 
        self.layer2 = self.build_cellblock(128, layers_dims[1], M, N, B[5:9], bias[5:9], W[5:9],stride=2,abc=False)
        self.layer3 = self.build_cellblock(256, layers_dims[2],M, N, B[9:13], bias[9:13], W[9:13],stride=2,abc=False)
        self.layer4 = self.build_cellblock(512, layers_dims[3],M, N, B[13:], bias[13:], W[13:],stride=2,abc=False)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        x=self.stem(inputs)
        x=self.stem2(x)
        # print(x.shape)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, M, N, B, bias, W,stride=1,abc=True):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, M, N, B[0:2], bias[0:2], W[0:2],stride,abc=abc)) 

        for i in range(1, blocks):    
            res_blocks.add(CellBlock(filter_num, M, N, B[i+1:i+3], bias[i+1:i+3], W[i+1:i+3],stride=1,abc=abc))

        return res_blocks


def build_ResNet(NetName, nb_classes,M, N, B, bias, W):
    ResNet_Config = {'ResNet18':[2,2,2,2], 
                    'ResNet34':[3,4,6,3]}

    return ResNet(ResNet_Config[NetName], nb_classes,M, N, B, bias, W) 

