#!/usr/bin/env python
# coding: utf-8

## Need to activate the python kernal of => python(v3.7.12)
# all the necessary packages are installed in this kernal will be provided at <requirements_benchmarking.txt>


import tensorflow as tf
import numpy as np
from scipy import io, sparse
from numpy import genfromtxt
from multiprocessing import Pool
tf.compat.v1.disable_v2_behavior()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import timeit
import pandas as pd

from tqdm import tqdm


from sklearn.neighbors import LSHForest


# Helper function to split data into chunks for LSH
def chunk(a, n):
    k, m = divmod(len(a), n)
    return list(tuple(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]) for i in range(n))


# # LSH Sampling function
# def lsh_main(data, obs):
#     global lshf
#     lshf = LSHForest(n_estimators=10, random_state=42)
#     lshf.fit(sparse.coo_matrix(data))
#     query_sets = chunk(range(obs), p)
#     with Pool(processes=p) as pool:
#         NN_set = list(tqdm(pool.imap(knn, query_sets), total=p, desc="Processing LSH chunks"))
#     indices = np.vstack(NN_set)
#     arr1 = np.ones(obs)
#     Nb = np.zeros(4)
#     m = np.zeros(4)
#     for i in range(obs):
#         if arr1[i] != 0:
#             Nb = indices[i][1:5]
#             arr1[Nb] = m
#     return arr1

def lsh_main(data,obs):
    global lshf 
    lshf = LSHForest(n_estimators=10, random_state=42)
    lshf.fit(sparse.coo_matrix(data)) 
    query_sets = chunk(range(obs),p)
    pool = Pool(processes=p)  
    NN_set = pool.map(knn, query_sets)
    pool.close()
    indices = np.vstack(NN_set)
    arr1=np.ones(obs)
    Nb=np.zeros(4)
    m=np.zeros(4)
    for i in range(0,obs):
        if arr1[i]!=0:
            Nb = indices[i][1:5]
            arr1[Nb]=m
    return arr1


# Nearest Neighbor search with LSH
def knn(q_idx):
    distances, indices = lshf.kneighbors(Xnew[q_idx, :], n_neighbors=5)
    return indices

## Muraro files don't need the index_col = 0 but need header = 0 
## CBMC needs index_col = 0 and header = 0 
## pollen needs header = None 
## yan needs header = None 

# Load and preprocess data
data = pd.read_csv('/path/to/the/real/data/muraro_expression_matrix.csv', header = 0)

print(data.shape)

# data = data.T

print(data.shape)

# data = data.values

n_samples, n_features = data.shape
x_plot = data.values
Xnew = x_plot
p = 20

row, col = data.shape


# LSH iteration (itr=1 for most cases, 2 for Klein)
itr = 1
for i in range(itr):
    rowlsh = Xnew.shape[0]
    result = lsh_main(Xnew, rowlsh)
    c = np.nonzero(result)
    c1 = c[0]
    Xnew = Xnew[c1, :]



tf.reset_default_graph()  # Crucial for preventing variable accumulation between runs

# Then disable v2 behavior
tf.disable_v2_behavior()




# Noise sampling function
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# Generator network
def generator(Z, hsize=[16, 16], reuse=tf.AUTO_REUSE):
    with tf.variable_scope("GAN/Generator", reuse=reuse):
        h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, col)
    return out

# Discriminator network
def discriminator(X, hsize=[16, 16], reuse=tf.AUTO_REUSE):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2, col)
        out = tf.layers.dense(h3, 1)
    return out, h3




# Placeholders
X = tf.placeholder(tf.float32, [None, col])
Z = tf.placeholder(tf.float32, [None, col])



# GAN components
G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse=True)



# Loss functions
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits, labels=tf.ones_like(r_logits)) +
                           tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits, labels=tf.ones_like(f_logits)))




# Variables and optimizers
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_vars)


# import torch

# torch.manual_seed(42)

tf.set_random_seed(42)



config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True)
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())



# Training parameters
nd_steps = 10
ng_steps = 10
batch_size = 64
num_batches = (row + batch_size - 1) // batch_size




# Training loop
start = timeit.default_timer()
for i in range(2001):
    # Generate Z_batch for this iteration
    da1 = sample_Z(row - Xnew.shape[0], col)
    Z_batch = np.row_stack((da1, Xnew))
    # Shuffle indices for real and generated data
    indices_X = np.random.permutation(row)
    indices_Z = np.random.permutation(row)
    
    # Discriminator training with mini-batches
    for _ in range(nd_steps):
        k = np.random.randint(0, num_batches)
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, row)
        X_mb = x_plot[indices_X[start_idx:end_idx], :]
        Z_mb = Z_batch[indices_Z[start_idx:end_idx], :]
        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_mb, Z: Z_mb})
    
    # Generator training with mini-batches
    for _ in range(ng_steps):
        k = np.random.randint(0, num_batches)
        start_idx = k * batch_size
        end_idx = min((k + 1) * batch_size, row)
        Z_mb = Z_batch[indices_Z[start_idx:end_idx], :]
        _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_mb})
    
    print("Iteration: %d\t Discriminator loss: %.4f\t Generator loss: %.4f" % (i, dloss, gloss))

stop = timeit.default_timer()
print('Time: ', stop - start)



######################## DATA GENERATION ########################

batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)


# Data generation

batch_sizes = (np.arange(0.25, 1.75, 0.25) * n_samples).astype(int)
for i, rows in enumerate(batch_sizes):
    row1 = batch_sizes[i]
    da1 = sample_Z(row1, col)
    Z_batch = da1
    synthetic_data = sess.run(G_sample, feed_dict={Z: Z_batch})
    with open(f"../path/to/save/the/data/lsh_muraro_generated_mixdata_iter{str(i)}.csv", "wb") as f:
        g_plot_pd = pd.DataFrame(synthetic_data)
        g_plot_pd.to_csv(f, index=False, header=False)
    print(g_plot_pd.shape)






