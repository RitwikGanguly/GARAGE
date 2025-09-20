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



latent_dim = 100
batch_size = 64




def encoder(x, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        h1 = tf.layers.dense(x, 32, activation=tf.nn.tanh)
        # Latent mean and log variance
        z_mean = tf.layers.dense(h1, latent_dim)
        z_logvar = tf.layers.dense(h1, latent_dim)
    return z_mean, z_logvar

# Reparameterization trick: sample from latent space
def sample_z(z_mean, z_logvar):
    eps = tf.random_normal(shape=tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_logvar) * eps
    return z

# Decoder network: mirror structure of the encoder with a single hidden layer
def decoder(z, reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        h1 = tf.layers.dense(z, 32, activation=tf.nn.tanh)
        # Linear activation for reconstruction (poorer quality for our benchmark)
        x_recon = tf.layers.dense(h1, n_features)
    return x_recon



tf.reset_default_graph()



x_input = tf.placeholder(tf.float32, shape=[None, n_features], name="x_input")

# Build the VAE: Encode, reparameterize, then decode
z_mean, z_logvar = encoder(x_input)
z = sample_z(z_mean, z_logvar)
x_reconstructed = decoder(z)



# Reconstruction loss (using mean squared error)
recon_loss = tf.reduce_mean(tf.square(x_input - x_reconstructed))
# KL divergence loss
kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
# Total loss is the sum of reconstruction and KL divergence
loss = recon_loss + kl_loss



optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(loss)



tf.set_random_seed(42)



config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.7),
    device_count={'GPU': 1},
    log_device_placement=False,
    intra_op_parallelism_threads=32,
    inter_op_parallelism_threads=32
)



sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


epochs = 2001
for epoch in range(epochs):
    idx = np.random.permutation(n_samples)
    for i in range(0, n_samples, batch_size):
        batch = data[idx[i:i+batch_size]]
        _, loss_val, rec, kl = sess.run([train_op, loss, recon_loss, kl_loss],
                                        feed_dict={x_input: batch})
        
    print("Epoch: {}  Loss: {:.4f}  Recon: {:.4f}  KL: {:.4f}".format(epoch, loss_val, rec, kl))



################################# DATA GENERATION ################################


batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)

for i, row1 in enumerate(batch_sizes):
    z_sample = np.random.normal(0, 1, (row1, latent_dim))
    x_gen = sess.run(x_reconstructed, feed_dict={z: z_sample})
    # Save generated synthetic data for later benchmarking (e.g., ARI score)
    pd.DataFrame(x_gen).to_csv(f"../path/to/save/the/data/vae_muraro_generated_mixdata_iter{str(i)}.csv", index=False, header=False)
    print(f"Iteration {i}: Generated data shape: {x_gen.shape}")

