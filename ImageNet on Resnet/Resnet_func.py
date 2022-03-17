#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Sequential,regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D




class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')

        self.conv2 = Conv2D(filter_num, (3,3), strides=1, padding='same')
        self.bn2 = BatchNormalization()

        if stride !=1:
            self.residual = Conv2D(filter_num, (1,1), strides=stride)
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

    def compute_weight(self):
        w1=self.conv1.weights[0]
        b1=self.conv1.weights[1]
        w2=self.conv2.weights[0]
        b2=self.conv2.weights[1]
        return w1,b1,w2,b2


class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes):
        super(ResNet, self).__init__()

        self.stem =Conv2D(64, (7,7), strides=(2,2),padding='same')
        self.stem2=Sequential([
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3,3), strides=(2,2), padding='same')
        ]) 

        self.layer1 = self.build_cellblock(64, layers_dims[0]) 
        self.layer2 = self.build_cellblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_cellblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_cellblock(512, layers_dims[3], stride=2)

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x=self.stem(inputs)
        # print(x.shape)
        x=self.stem2(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        
        x=self.avgpool(x)
        x=self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride)) 

        for _ in range(1, blocks):    
            res_blocks.add(CellBlock(filter_num, stride=1))

        return res_blocks
    def get_weights(self):
        W_list=[]
        W_list.append(self.stem.get_weights())
        for l in self.layer1.layers:
            W_list.append(l.compute_weight())
        for l in self.layer2.layers:
            W_list.append(l.compute_weight())
        for l in self.layer3.layers:
            W_list.append(l.compute_weight())
        for l in self.layer4.layers:
            W_list.append(l.compute_weight()) 
        return W_list
            
                  
def build_ResNet(NetName, nb_classes):
    ResNet_Config = {'ResNet18':[2,2,2,2], 
                'ResNet34':[3,4,6,3]}

    return ResNet(ResNet_Config[NetName], nb_classes) 





