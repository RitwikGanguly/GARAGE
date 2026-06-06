#!/usr/bin/env python
# coding: utf-8

## Need to activate the python kernal of => python(v3.7.12)
# all the necessary packages are installed in this kernal will be provided at <requirements_benchmarking.txt>


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd


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



n_samples, n_features = data.shape
latent_dim = 100
batch_size = 64



config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(
        allow_growth=False,  # Disable incremental allocation
        per_process_gpu_memory_fraction=0.7  # Allocate 70% of 48GB = ~33.6GB
    ),
    device_count={'GPU': 1},
    log_device_placement=False,  # Disable verbose logging
    intra_op_parallelism_threads=32,  # Utilize more CPU threads
    inter_op_parallelism_threads=32
)



def generator(z, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('generator', reuse=reuse):
        # Single hidden layer with 32 neurons and tanh activation
        h1 = tf.layers.dense(z, 32, activation=tf.nn.tanh)
        # Output layer (linear activation) to produce n_features values
        output = tf.layers.dense(h1, n_features)
    return output

# ---------------------------
# Define the Discriminator Network (Default Parameters)
# ---------------------------
def discriminator(x, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Single hidden layer with 32 neurons and leaky ReLU activation
        h1 = tf.layers.dense(x, 32, activation=tf.nn.leaky_relu)
        # Output logits layer (linear activation)
        logits = tf.layers.dense(h1, 1)
    return logits



tf.reset_default_graph()  # Crucial for preventing variable accumulation between runs

# Then disable v2 behavior
tf.disable_v2_behavior()



real_input = tf.placeholder(tf.float32, shape=[None, n_features], name='real_input')
z_input = tf.placeholder(tf.float32, shape=[None, latent_dim], name='z_input')

# ---------------------------
# Build the GAN Networks
# ---------------------------
gen_sample = generator(z_input)
disc_real = discriminator(real_input)
disc_fake = discriminator(gen_sample, reuse=True)

# ---------------------------
# Define Loss Functions using Sigmoid Cross-Entropy
# ---------------------------
# Discriminator losses on real and fake data
disc_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
disc_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
disc_loss = disc_loss_real + disc_loss_fake

# Generator loss tries to fool the discriminator
gen_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))

# ---------------------------
# Optimizers (Default Adam with Learning Rate = 0.0001)
# ---------------------------
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

gen_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
disc_opt = tf.train.AdamOptimizer(learning_rate=0.0001)

gen_train = gen_opt.minimize(gen_loss, var_list=gen_vars)
disc_train = disc_opt.minimize(disc_loss, var_list=disc_vars)



tf.set_random_seed(42)



sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# Training loop
for epoch in range(2001):
    idx = np.random.permutation(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch = data[idx[i:i+batch_size]]
        noise = np.random.normal(0, 1, (len(batch), latent_dim))
        
        # Train discriminator first then generator
        _, d_loss = sess.run([disc_train, disc_loss],
                             {real_input: batch, z_input: noise})
        _, g_loss = sess.run([gen_train, gen_loss],
                             {z_input: noise})
        
    print("Iteration: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" 
            % (epoch, d_loss, g_loss))



################### DATA GENERATION ###################


batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)

for i, row1 in enumerate(batch_sizes):
    Z_batch = np.random.normal(0, 1, (row1, latent_dim))
    
    # Generate synthetic data using the trained generator
    g_plot = sess.run(gen_sample, feed_dict={z_input: Z_batch})
    
    # Save generated data
    pd.DataFrame(g_plot).to_csv(f"../path/to/save/the/data/gan_muraro_generated_mixdata_iter{str(i)}.csv", index=False, header=False)
    
    print(f"Iteration {i}: Generated data shape: {g_plot.shape}")

