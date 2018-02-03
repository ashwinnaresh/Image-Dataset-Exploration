#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:21:44 2018

@author: ashwin
"""

import numpy as np
import cPickle

def unpickle(file):
    path = 'data/cifar-10-batches-py/'
    with open(path + file, 'rb') as file_open:
        data = cPickle.load(file_open)
    return data

if __name__ == '__main__':
    # preprocess the data
    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x1d_train = []
    x1d_test = []
    dim = 32
    count = 0
    
    for f in files:
        data_dict = unpickle(f)
        data = data_dict['data']/255.
        
        labels = data_dict['labels']
        images = []
        for img in data:
            new_img = np.zeros((dim,dim,3))
            r = img[:(dim * dim)].reshape((dim,dim))
            b = img[(dim * dim):2 * (dim * dim)].reshape((dim,dim))
            g = img[2 * (dim * dim):].reshape((dim,dim))
            
            for i in range(dim):
                for j in range(dim):
                    new_img[i][j] = np.array([r[i][j]] + [b[i][j]] + [g[i][j]])

            images.append(new_img)
           
        if 'test' not in f:
            x_train.extend(np.array(images))
            y_train.extend(np.array(labels))
            x1d_train.extend(data.tolist())
            
        else:
            x_test.extend(np.array(images))
            y_test.extend(np.array(labels))
            x1d_test.extend(data.tolist())
            
        count += 1
        print count
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print 'x_train shape : ', x_train.shape
    print 'y_train shape : ', y_train.shape
    print 'x_test shape : ', x_test.shape
    print 'y_test shape : ', y_test.shape
    
    np.save(open('data/cifar_10_x_train.npy','w'), x_train)
    np.save(open('data/cifar_10_y_train.npy','w'), y_train)
    np.save(open('data/cifar_10_x_test.npy','w'), x_test)
    np.save(open('data/cifar_10_y_test.npy','w'), y_test)
        
    x1d_train = np.array(x1d_train)
    x1d_test = np.array(x1d_test)
    print 'x_1dtrain shape : ', x1d_train.shape
    print 'x_1dtrain shape : ', x1d_test.shape
    
    np.save(open('data/cifar_10_x1d_train.npy','w'), x1d_train)
    np.save(open('data/cifar_10_x1d_test.npy','w'), x1d_test)
    
                        
     