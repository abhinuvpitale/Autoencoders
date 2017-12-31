import tensorflow as tf
import numpy as np
import os
import sys

# Get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Data',one_hot=True)


lib_path = os.path.abspath(path='.')
sys.path.append(lib_path)

import parameters as par
import layers


x_input = tf.placeholder(tf.float32, shape=[par.batch_size, par.n_input],name = 'Input')
x_target = tf.placeholder(tf.float32, shape=[par.batch_size, par.n_input],name='Target')
decoder_output = tf.placeholder(tf.float32, shape=[1, par.z_dim], name='DecoderInput')


def train(train_model):
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = layers.encoder(x_input)
        decoder_output = layers.decoder(encoder_output)
    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = layers.decoder(decoder_input, reuse=True)

    loss = tf.reduce_mean(tf.square(decoder_output-decoder_image))

    optimiser = tf.train.AdamOptimizer(learning_rate=par.learning_rate, beta1=par.beta1).minimize(loss)
    init = tf.global_variables_initializer()

    # Viz
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    input_images = tf.reshape(x_input,[-1,28,28,1])
    


