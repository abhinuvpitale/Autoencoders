import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime


# Get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./Data',one_hot=True)

# Adds path to the sys_path to make sure our libraries are included.
lib_path = os.path.abspath(path='.')
sys.path.append(lib_path)

import parameters as par
import layers

x_input = tf.placeholder(tf.float32, shape=[par.batch_size, par.n_input],name = 'Input')
x_target = tf.placeholder(tf.float32, shape=[par.batch_size, par.n_input],name='Target')
decoder_input = tf.placeholder(tf.float32, shape=[1, par.z_dim], name='Decoder_input')

# Generates images from the trained model. uses input between 0-1 as input.
def generate_image_grid(sess, op):
    nx,ny = 1,1
    plt.subplot()
    # Generates
    gs = gridspec.GridSpec(nx,ny,hspace=0.05,wspace=0.05)

    for i, g in enumerate(gs):
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        z = np.reshape(z, (1, 2))
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()

# used to create the appropriate paths for saving all the data.
def form_results():
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_autoencoder".format(datetime.date.today(), par.z_dim, par.learning_rate, par.batch_size, par.n_epochs, par.beta1).replace(':','').replace('.','').replace(' ','').replace('-','_')
    tensorboard_path = par.results_path + folder_name + '/Tensorboard'
    saved_model_path = par.results_path + folder_name + '/Saved_models/'
    log_path = par.results_path + folder_name + '/log'
    if not os.path.exists(par.results_path + folder_name):
        os.mkdir(par.results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

# Start the training here.
def train(train_model):
    with tf.variable_scope(tf.get_variable_scope()):
        # Creates the model, sends the input thorugh the Encoder
        encoder_output = layers.encoder(x_input)
        # Send's the encoder's output via the Decoder -> Should result in perfect reconstruction
        decoder_output = layers.decoder(encoder_output)
    with tf.variable_scope(tf.get_variable_scope()):
        # Sends the random sequence via the Decoder
        decoder_image = layers.decoder(decoder_input, reuse=True)

    # Mean Square Loss funciton
    loss = tf.reduce_mean(tf.square(x_target-decoder_output))

    # Adam Optimiser - beta1 - exponential decay rate for the 1st moment
    optimiser = tf.train.AdamOptimizer(learning_rate=par.learning_rate, beta1=par.beta1).minimize(loss)
    # Duh
    init = tf.global_variables_initializer()

    # Viz
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    # Get the right shape back
    input_images = tf.reshape(x_input,[-1,28,28,1])
    # See the output images
    generated_images = tf.reshape(decoder_output,[-1,28,28,1])
    tf.summary.image(name='Input Images',tensor=input_images,max_outputs=10)
    tf.summary.image(name='Generated Images',tensor=generated_images,max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Model saving
    saver = tf.train.Saver()
    step = 0

    # Training the model
    with tf.Session() as sess:
        sess.run(init)
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            # creates the main tensorboard file, where interactive display can be shown. Passing the graph is the constructor adds it to the viz.
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for i in range(par.n_epochs):
                # Create batches
                n_batches = int(mnist.train.num_examples/par.batch_size)
                for b in range(n_batches):
                    batch_x, _ = mnist.train.next_batch(par.batch_size)
                    # input and target are the same image as this is auto-encoder
                    sess.run(optimiser,feed_dict={x_input:batch_x,x_target:batch_x})
                    if b%par.display_step == 0 or b == 1:
                        # get the summary and the loss.
                        batch_loss,summary = sess.run([loss,summary_op],feed_dict={x_input:batch_x,x_target:batch_x})
                        # summary_op adds the summary to the tensorboard
                        writer.add_summary(summary,global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Loss: {}".format(batch_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Loss: {}\n".format(batch_loss))
                    step += 1
                    # save the model for each epoch.
                saver.save(sess, save_path=saved_model_path, global_step=step)
            print("Tensorboard Path: {}".format(tensorboard_path))
            print("Log Path: {}".format(log_path + '/log.txt'))
            print("Saved Model Path: {}".format(saved_model_path))
        else:
            # get result path
            all_results = os.listdir(par.results_path)
            all_results.sort()
            # get the last trained model
            saver.restore(sess,save_path=tf.train.latest_checkpoint(par.results_path+'/'+all_results[-1]+'/Saved_models'))
            # generate the grid to view the images.
            generate_image_grid(sess,op=decoder_image)

train(train_model=True)
