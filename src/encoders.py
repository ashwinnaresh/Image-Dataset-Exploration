#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:00:34 2018

@author: ashwin
"""
from keras.layers import Input, Dense, Lambda
from keras.models import load_model
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np
import sys 

from tsne_visualization import visualize

def RAE(in_dim, out_dim):
    # this is the size of our encoded representations
    encoding_dim = out_dim
    
    # this is our input placeholder
    input_img = Input(shape=(in_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(in_dim, activation='sigmoid')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    return autoencoder, encoder, decoder

def VAE(batch_size,original_dim,intermediate_dim,latent_dim,epsilon_std=1.,data='cifar'):
    def vae_loss(x, x_decoded_mean):
        # loss function for VAE
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    x = Input(shape=(original_dim,))
    if data == 'cifar':
        h = Dense(intermediate_dim, activation='relu')(x)
    else:
        # this change is required, otherwise, we get loss as nan
        h = Dense(intermediate_dim, activation='tanh')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    # instantiate VAE model
    vae = Model(x, x_decoded_mean)
    
    vae.compile(optimizer='rmsprop',loss=vae_loss)
    vae.summary()
    
    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)
    
    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    return vae, encoder, generator


if __name__ == '__main__':
    # load the data
    if sys.argv[1] == 'cifar':
        x_train = np.load(open('data/cifar_10_x1d_train.npy'))
        y_train = np.load(open('data/cifar_10_y_train.npy'))
        x_test = np.load(open('data/cifar_10_x1d_test.npy'))
        y_test = np.load(open('data/cifar_10_y_test.npy'))
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    print 'Input shape : ', x_train.shape
    # build the RAE 
    rae, encoder, decoder = RAE(x_train.shape[1],32)
    # train RAE
    rae.fit(x_train,x_train,batch_size=32,epochs=10,validation_data=(x_test,x_test))
    rae.save('rae.h5')
    # test RAE
    rae = load_model('rae.h5')
    # store RAE representations
    rae_embedding = encoder.predict(x_test)
    print 'RAE Embedding shape : ', rae_embedding.shape
    # t-SNE on RAE embedding
    if sys.argv[1] == 'cifar':
        visualize(rae_embedding[:2000],y_test[:2000],data='cifar')
    else:
        visualize(rae_embedding[:2000],y_test[:2000],data='mnist')
    # build the VAE 
    if sys.argv[1] == 'cifar':
        vae, encoder, generator = VAE(64,x_train.shape[1],256,2,data='cifar')
    else:
        vae, encoder, generator = VAE(64,x_train.shape[1],256,2,data='mnist')
    # train VAE
    vae.fit(x_train,x_train, shuffle=True, epochs=10, batch_size=64, validation_data=(x_test, x_test))
    vae.save_weights('vae.h5')
    # test VAE
    vae.load_weights('vae.h5')
    # store VAE representations 
    vae_embedding = encoder.predict(x_test,batch_size=32)
    print vae_embedding.shape
    # t-SNE on VAE embedding
    if sys.argv[1] == 'cifar':
        visualize(vae_embedding[:2000],y_test[:2000],data='cifar')
    else:
        visualize(vae_embedding[:2000],y_test[:2000],data='mnist')
    