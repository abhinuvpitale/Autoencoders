import tensorflow as tf
import os
import sys


lib_path = os.path.abspath(path='.')
sys.path.append(lib_path)

import parameters as par


# Creates generic dense layer, to be used in encoder and decoder functions!
def dense(x, input, output, name):
    with tf.variable_scope(name,reuse=None):
        weights = tf.get_variable("weights",shape=[input,output], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        biases = tf.get_variable("biases",shape=[output], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x,weights),biases,name='matmul')
        return out


# Encoder - Outputs the latent variable output, which is the compressed form of the input.
def encoder(x,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x,par.n_input,par.n_l1,'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, par.n_l1, par.n_l2, 'e_dense_1'))
        latent_variable = dense(e_dense_2,par.n_l2,par.z_dim,'e_latent_variable')
        return latent_variable


# Decoder  -Outputs an image from the latent variable
def decoder(x,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(x,par.z_dim,par.n_l2,'d_dense_1')
        d_dense_2 = tf.nn.relu(d_dense_1,par.n_l2,par.n_l1,'d_dense_2')
        output = tf.nn.sigmoid(dense(d_dense_2,par.n_l1,par.n_input,'d_output'))
        return output
