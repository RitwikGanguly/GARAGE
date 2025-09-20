#!/usr/bin/env python
# coding: utf-8

## Need to activate the python kernal of => python(v3.7.12)
# all the necessary packages are installed in this kernal will be provided at <requirements_benchmarking.txt>

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import os


## Muraro files don't need the index_col = 0 but need header = 0 
## CBMC needs index_col = 0 and header = 0 
## pollen needs header = None 
## yan needs header = None 


# Load and preprocess data
data = pd.read_csv('/path/to/the/real/data/muraro_expression_matrix.csv', header = 0)

print(data.shape)

# data = data.T

print(data.shape)

data = data.values

n_samples, n_features = data.shape
x_plot = data
Xnew = x_plot
p = 20

row, col = data.shape



latent_dim = 100
batch_size = 64
n_critic = 5  # Update critic 5 times for each generator update
clip_value = 0.01  # Weight clipping range
learning_rate = 0.0005  



def generator(z, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("generator", reuse=reuse):
        h1 = tf.compat.v1.keras.layers.Dense(32, activation=tf.nn.tanh)(z)
        output = tf.compat.v1.keras.layers.Dense(n_features)(h1)
    return output

# ---------------------------
# Define the Critic (Discriminator)
# ---------------------------
def critic(x, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("critic", reuse=reuse):
        h1 = tf.compat.v1.keras.layers.Dense(32, activation=tf.nn.tanh)(x)
        score = tf.compat.v1.keras.layers.Dense(1)(h1)
    return score



tf.reset_default_graph()  # Crucial for preventing variable accumulation between runs

# Then disable v2 behavior
tf.disable_v2_behavior()



real_input = tf.placeholder(tf.float32, shape=[None, n_features], name='real_input')
z_input = tf.placeholder(tf.float32, shape=[None, latent_dim], name='z_input')



gen_sample = generator(z_input)
critic_real = critic(real_input)
critic_fake = critic(gen_sample, reuse=True)



# Critic loss: maximize D(real) - D(fake)
critic_loss = -tf.reduce_mean(critic_real) + tf.reduce_mean(critic_fake)
# Generator loss: maximize D(fake) i.e. minimize -D(fake)
gen_loss = -tf.reduce_mean(critic_fake)



gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

# Use RMSProp optimizer (as recommended in the original paper)
gen_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
critic_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

gen_train = gen_opt.minimize(gen_loss, var_list=gen_vars)
critic_train = critic_opt.minimize(critic_loss, var_list=critic_vars)



# Weight Clipping for Critic Variables
# ---------------------------
clip_ops = []
for var in critic_vars:
    clip_ops.append(tf.assign(var, tf.clip_by_value(var, -clip_value, clip_value)))
clip_critic = tf.group(*clip_ops)



import torch

torch.manual_seed(42)



config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
        allow_growth=True,
        visible_device_list="0",
        per_process_gpu_memory_fraction=0.17  # ~8GB out of 48GB
    ),
    intra_op_parallelism_threads=32,
    inter_op_parallelism_threads=32,
    device_count={'GPU': 1},
    log_device_placement=False
)



try:
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print("Session created successfully.")
except Exception as e:
    print("Error creating session:", e)



n_epochs = 2001

for epoch in range(n_epochs):
    idx = np.random.permutation(n_samples)
    
    # Update critic n_critic times
    for i in range(0, n_samples, batch_size):
        batch = data[idx[i:i+batch_size]]
        noise = np.random.normal(0, 1, (len(batch), latent_dim))
        # Train critic
        sess.run(critic_train, feed_dict={real_input: batch, z_input: noise})
        # Apply weight clipping to critic
        sess.run(clip_critic)
        
    # Generator update (using one batch of noise)
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    sess.run(gen_train, feed_dict={z_input: noise})
    
    c_loss, g_loss = sess.run([critic_loss, gen_loss],
                                feed_dict={real_input: batch, z_input: noise})
    print("Epoch: {} || Critic Loss: {:.4f} || Generator Loss: {:.4f}".format(epoch, c_loss, g_loss))




####################### DATA GENERATION #######################


batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)
for i, rows in enumerate(batch_sizes):
    Z_batch = np.random.normal(0, 1, (rows, latent_dim))
    synthetic_data = sess.run(gen_sample, feed_dict={z_input: Z_batch})
    pd.DataFrame(synthetic_data).to_csv(f"../path/to/save/the/data/wgan_muraro_generated_mixdata_iter{str(i)}.csv", index=False, header=False)
    print("Iteration {}: Generated data shape: {}".format(i, synthetic_data.shape))




