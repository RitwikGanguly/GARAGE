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



# Model parameters
latent_dim = 100
batch_size = 64
n_critic = 5        # Number of critic updates per generator update
learning_rate = 5e-4  # Lower learning rate for stability
lambda_lagrange = 10 # Constraint penalty weight



def generator(z, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        # Hidden layer with 128 neurons and LeakyReLU activation
        h1 = tf.layers.dense(z, 128, activation=tf.nn.leaky_relu)
        # Output layer (linear) to match the number of features
        output = tf.layers.dense(h1, n_features)
    return output

# ---------------------------
# Define the fGAN Discriminator Network
# ---------------------------
def f_discriminator(x, reuse=None):
    with tf.variable_scope("f_discriminator", reuse=reuse):
        # Hidden layer with 128 neurons and LeakyReLU activation
        h1 = tf.layers.dense(x, 128, activation=tf.nn.leaky_relu)
        # Output a single scalar value T(x)
        T = tf.layers.dense(h1, 1)
    return T




tf.reset_default_graph()



real_input = tf.placeholder(tf.float32, shape=[None, n_features], name='real_input')
z_input = tf.placeholder(tf.float32, shape=[None, latent_dim], name='z_input')

# Build generator and fGAN discriminator networks
G_sample = generator(z_input)
T_real = f_discriminator(real_input)
T_fake = f_discriminator(G_sample, reuse=True)



mean_real = tf.reduce_mean(T_real)
mean_fake = tf.reduce_mean(T_fake)
var_real = tf.reduce_mean(tf.square(T_real - mean_real))
var_fake = tf.reduce_mean(tf.square(T_fake - mean_fake))




# Fisher ratio calculation
epsilon = 1e-8
fisher_ratio = (mean_real - mean_fake) / tf.sqrt(var_real + var_fake + epsilon)



constraint = tf.reduce_mean(tf.square(T_real) + tf.square(T_fake)) - 1.0
penalty = tf.square(tf.maximum(constraint, 0.0))



# Final losses
D_loss = -fisher_ratio + lambda_lagrange * penalty
G_loss = -fisher_ratio



gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='f_discriminator')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
D_train_op = optimizer.minimize(D_loss, var_list=disc_vars)
G_train_op = optimizer.minimize(G_loss, var_list=gen_vars)



tf.set_random_seed(42)



config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True)
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# for epoch in range(2001):
#     # Update critic multiple times
#     for _ in range(n_critic):
#         batch = data[np.random.choice(n_samples, batch_size)]
#         noise = np.random.normal(0, 1, (batch_size, latent_dim))
#         sess.run(D_train_op, feed_dict={real_input: batch, z_input: noise})
    
#     # Update generator
#     noise = np.random.normal(0, 1, (batch_size, latent_dim))
#     sess.run(G_train_op, feed_dict={z_input: noise})
    
#     c_loss, g_loss = sess.run([D_loss, G_loss],
#                             feed_dict={real_input: batch, z_input: noise})
#     print(f"Epoch {epoch} | Critic Loss: {c_loss:.4f} | Gen Loss: {g_loss:.4f}")



n_epochs = 2001

for epoch in range(n_epochs):
    idx = np.random.permutation(n_samples)
    
    # Create batches
    batches = [data[idx[i:i+batch_size]] for i in range(0, n_samples, batch_size)]
    
    for batch in batches:
        current_batch_size = len(batch)
        noise = np.random.normal(0, 1, (current_batch_size, latent_dim))
        
        # Update discriminator with both real and fake data
        _, d_loss = sess.run([D_train_op, D_loss], 
                            feed_dict={real_input: batch, z_input: noise})
        
        # Update generator with BOTH real and noise inputs
        _, g_loss = sess.run([G_train_op, G_loss],
                            feed_dict={real_input: batch, z_input: noise})
    
    # Print progress using last batch
    print(f"Epoch {epoch} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")


####################### DATA GENERATION #######################


batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)
for i, rows in enumerate(batch_sizes):
    noise = np.random.normal(0, 1, (rows, latent_dim))
    synthetic_data = sess.run(G_sample, feed_dict={z_input: noise})
    pd.DataFrame(synthetic_data).to_csv(f"../path/to/save/the/data/fgan_muraro_generated_mixdata_iter{str(i)}.csv", index=False, header=False)
    print("Iteration {}: Generated data shape: {}".format(i, synthetic_data.shape))



