#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:47:20 2018

@author: ashwin
"""

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
import sys

def build_network(x_train):
    # build the CNN
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    if sys.argv[1] == 'cifar':
        x_train = np.load(open('data/cifar_10_x_train.npy'))
        y_train = np.load(open('data/cifar_10_y_train.npy'))
        x_test = np.load(open('data/cifar_10_x_test.npy'))
        y_test = np.load(open('data/cifar_10_y_test.npy'))
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
    
    x_train /= 255.
    x_test /= 255.
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    print 'Training Input shape : ', x_train.shape
    print 'Training Target shape : ', y_train.shape
    print 'Training Input shape : ', x_test.shape
    print 'Training Target shape : ', y_test.shape
    batch_size = 32
    epochs = 5
    model = build_network(x_train)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,validation_data=(x_test,y_test))
    open('hist.p','w').write(str(history))
    model.save('cifar10_cnn.h5')
    load_model('cifar10_cnn.h5')
    loss = model.evaluate(x_train, y_train,
              batch_size=batch_size)
    print loss
    
    plt.plot(range(epochs),history.history['loss'],'b',range(epochs),history.history['val_loss'],'g')
    plt.show()