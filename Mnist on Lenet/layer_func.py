#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from sklearn import linear_model 
from tensorflow.keras import initializers
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
# In[ ]:


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
"""
    B_reshape=tf.reshape(B,[-1,M]).eval()
    W_reshape=tf.reshape(W,[-1,1]).eval()
    
    variables=[alpha]
    for e in range(num_epoch):
        with tf.GradientTape() as tape:
            B = tf.cast(B,tf.float32)
            W_pred = alpha@B
            loss = tf.keras.losses.MSE(W, W_pred)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,variables))
    return alpha,loss,op
"""
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
def activation_and_conv_sigmoid(X,beta,v,M,N,B,alpha,bias,stride):
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
    A0=tf.sign(tf.sigmoid(X+v[0])-0.5)
    ConvWR = tf.zeros_like(ApproxConv(B,A0, alpha,bias[0],M,stride))
    for n in range(N):
        An = tf.sign(tf.sigmoid(X+v[n])-0.5)
        ConvWR = tf.add(ConvWR, tf.multiply(ApproxConv(B,An,alpha,bias[n],M,stride),beta[n]))
    return ConvWR 

# In[ ]:

"""
class ABCLayer(tf.keras.layers.Layer):
    def __init__(self, M, N, B1, bias1, W1):
        super(ABCLayer, self).__init__()
        self.M = M
        self.N = N
        self.B1 = tf.identity(B1)
        self.bias1 = tf.identity(bias1)
        self.W1 = tf.identity(W1)
        self.alpha = tf.Variable(tf.random.normal(shape=(M,1)))
        #self.optimizer = tf.keras.optimizers.SGD() 
    def build(self, input_shape):
        self.beta = self.add_weight("beta",
                                    shape = [self.N])
        self.v = self.add_weight("v",
                                shape = [self.N])
        
        initializer_B = tf.constant_initializer(np.array(self.B1))
        self.B = self.add_weight("B",
                                shape = list(self.B1.shape),
                                initializer = initializer_B)
        
        initializer_bias = tf.constant_initializer(np.array(self.bias1))
        self.bias = self.add_weight("bias",
                                   shape = list(self.bias1.shape),
                                   initializer = initializer_bias)
        
        initializer_W = tf.constant_initializer(np.array(self.W1))
        self.W = self.add_weight("W",
                                shape = list(self.B1.shape)[1:],
                                initializer = initializer_W)
        super(ABCLayer, self).build(input_shape)
    def call(self, inputs):
        Z= get_binary_filter(self.W,self.M)
        self.B=Z[0]
        Y=optimize_alpha(self.W,self.B,self.M)
        self.alpha=Y[0]
        return activation_and_conv(inputs,self.beta,self.v,self.M,self.N,self.B,self.alpha,self.bias)
"""
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

class ABCLayer_sigmoid(tf.keras.layers.Layer):
    def __init__(self, M, N, B1, bias1, W1,stride):
        tf.config.run_functions_eagerly(True)
        super(ABCLayer_sigmoid, self).__init__()
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
        super(ABCLayer_sigmoid, self).build(input_shape)
    def call(self, inputs):
        Z= get_binary_filter(self.W,self.M)
        self.B=Z[0]
        Y=optimize_alpha(self.W,self.B,self.M)
        self.alpha=Y[0]
        return activation_and_conv_sigmoid(inputs,self.beta,self.v,self.M,self.N,self.B,self.alpha,self.bias,self.stride)


