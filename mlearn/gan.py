"""
Simple GAN 

Simplified version of http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
"""

import numpy as np
import tensorflow as tf

NUM_STEPS = 1000
BATCH_SIZE = 12
HIDDEN_SIZE = 4
LR = 0.01
MU = 4
SIGMA = 0.5

get_data_distribution = lambda n: np.random.normal(mu, sigma, n)
get_generator_distribution = lambda n, limit: np.linspace(-limit, limit, n) + np.random.random(n) * 0.01

def linear(data, output_dim):
    w = tf.Variable(tf.random_normal(shape=(int(data.get_shape()[1]), output_dim), name='w'))
    b = tf.Variable(tf.ones(shape=(output_dim), name = 'b'))
    return tf.matmul(data, w) + b
 
def generator(data, h_dim):
    h0 = tf.nn.softplus(linear(data, h_dim))
    h1 = linear(h0, 1)
    return h1

def discriminator(data, hidden_units):
    h0 = tf.tanh(linear(data, hidden_units * 2))
    h1 = tf.tanh(linear(h0, hidden_units * 2))
    h2 = tf.tanh(linear(h1, hidden_units * 2))
    h3 = tf.sigmoid(linear(h2, 1))
    return h3

def optimizer(loss, var_list):
    return tf.train.GradientDescentOptimizer(LR).minimize(loss, var_list=var_list)

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1))
    G = generator(z, HIDDEN_SIZE)

with tf.variable_scope('D') as scope:
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 1))
    D1 = discriminator(x, HIDDEN_SIZE)
    scope.reuse_variables()
    D2 = discriminator(G, HIDDEN_SIZE)

loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

opt_d = optimizer(loss_d, d_params)
opt_g = optimizer(loss_g, g_params)

with tf.Session() as session:
    tf.initialize_all_variables().run()

    for step in range(NUM_STEPS):
        # update discriminator
        x_samples = get_data_distribution(BATCH_SIZE)
        z_samples = get_generator_distribution(BATCH_SIZE, 8)
        step_loss_d, _ = session.run([loss_d, opt_d], {
            x: np.reshape(x_samples, (BATCH_SIZE, 1)),
            z: np.reshape(z_samples, (BATCH_SIZE, 1))
        })

        # update generator
        z_samples = get_generator_distribution(BATCH_SIZE, 8)
        step_loss_g, _ = session.run([loss_g, opt_g], {
            z: np.reshape(z_samples, (BATCH_SIZE, 1))
        })

        print('{}. Discriminator: {}\t Generator: {}'.format(step, step_loss_d, step_loss_g))
