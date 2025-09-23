#!/usr/bin/env python
# coding: utf-8
## Need to activate the conda env 

## importing the libraries

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
import timeit
import math
import pyreadr

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv





## Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## GAT - SubSampling

def gat_main(data,label,index_list, k):

    # import torch.nn.functional as F
    
    

    




    # Assuming attention1[0] has shape [num_heads, num_edges]
    # For example, if num_heads=2 and num_edges=450
    attention_coefficients = attention2[0]

    # Assuming edge_index has shape [2, num_edges]
    # edge_index[0] and edge_index[1] represent the source and target nodes of each edge
    edge_index_source_nodes = edge_index[0]

    # Convert X from a NumPy array to a PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32, device=edge_index.device)

    # Calculate attention weights for each node
    num_nodes = X.shape[0]

    # Initialize attention_weights as a PyTorch tensor
    attention_weights = torch.zeros(num_nodes, device=X.device)

    for i in range(num_nodes):
        # Find indices of edges incident to node i
        incident_edges_mask = edge_index_source_nodes == i
        incident_edges_attention = attention_coefficients[:, incident_edges_mask]

        # Convert the attention coefficients to float32 dtype
        incident_edges_attention = incident_edges_attention.float()

        # Aggregate attention coefficients for node i
        node_attention = torch.mean(incident_edges_attention, dim=1)
        attention_weights[i] = torch.mean(node_attention)

   # print("Attention Weights for Each Node:", attention_weights)

    sorted_indices = torch.argsort(attention_weights, descending=True)

    # Select the top-k nodes
    top_node=k
    
    top_k_nodes = sorted_indices[:top_node]

#    print("Top", k, "Nodes:", top_k_nodes)

    return(top_k_nodes)





## Muraro files don't need the index_col = 0 but need header = 0 (both df1, df2)
## CBMC needs index_col = 0 and header = 0 (both df1, df2)
## pollen needs header = None (both df1, df2)
## yan needs header = None (both df1, df2)

df2 = pd.read_csv(r"data/cell_types/yan_celltype.csv", header = None) 
df1 = pd.read_csv("data/expression_matrix", header=None)


## "YAN" & "CBMC" needs ==> transpose, "POLLEN" & "MURARO" does not need transpose


df1 = df1.T


df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)




df2 = df2.rename(columns={df2.columns[0]: "cell_type"})
df3 = pd.concat([df1, df2], axis=1)





n_sample, n_features = df1.shape






## for cbmc = 200, yan = 10, pollen = 25, muraro = 200

class_labels = df3.cell_type.value_counts()
rare_types = class_labels[class_labels <= 10].index.tolist()
selected_rows = df3.loc[df3['cell_type'].isin(rare_types)]
index_list = selected_rows.index.tolist()



# Encode labels
label_encoder = LabelEncoder()
df3['class_label_encoded'] = label_encoder.fit_transform(df3['cell_type'])
X = df3.iloc[:, :-2].values  # Features
y = df3['class_label_encoded'].values  # Encoded labels



# Hyperparameters
## here the percentage of leakage is multiplied by n_sample to get the actual leakage nodes
## as example 20% leakage, means multiplied by 0.2


k = math.ceil(n_sample*0.2)   # Top-k nodes



## Checking for NO LEAKAGE
### taking no node from GAT 
## to check the comparitive study

k = 0


x_plot = X
row=x_plot.shape[0] 
col=x_plot.shape[1]



## The Gat training and getting the top-k indices
Xnew_gat_indices = gat_main(X, y, index_list, k)



Xnew = x_plot[Xnew_gat_indices.numpy(), :]


# ## GAN - Data Generation

# ### NOISE




# Training parameters
nd_steps = 5
ng_steps = 2


torch.manual_seed(42)
np.random.seed(42)


# ## GARAGE Part


for i in range(20001):
    # Prepare X_batch (real data)
    X_batch_np = x_plot
    X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(device)
    # Prepare Z_batch (noise + selected real data features)
    row1 = row - Xnew.shape[0]
    if row1 < 0: row1 = 0 # Ensure non-negative for sample_Z if Xnew is larger than row

    da1 = sample_Z(row1, col)

    # Ensure Xnew is 2D even if it has 1 row after GAT selection
    current_xnew_np = Xnew
    if current_xnew_np.ndim == 1:
        current_xnew_np = current_xnew_np.reshape(1, -1)


    Z_batch_np = np.vstack((da1, current_xnew_np))


    Z_batch = torch.tensor(Z_batch_np, dtype=torch.float32).to(device)

    dloss_epoch = 0
    # Train Discriminator
    for _ in range(nd_steps):
        disc_optimizer.zero_grad()
        
        
        
        # Real samples
        real_output, r_rep = discriminator_pt(X_batch)
        
        fake_samples = generator_pt(Z_batch).detach() 
        fake_output, g_rep_dstep = discriminator_pt(fake_samples)
        
        ## Label Smoothning - without assigning direct 1 to real and 0 to fake, assign a fraction
        ## of label to the real and fake sample as 0.9 and 0.1 respectively.
        
        real_labels = torch.full_like(real_output, 0.9).to(device) 
        fake_labels = torch.full_like(fake_output, 0.1).to(device)
        # disc_loss_real = criterion_gan(real_output, torch.ones_like(real_output).to(device))
        
        disc_loss_real = criterion_gan(real_output, real_labels)
        
        # Fake samples
        # Use Z_batch_for_g which is guaranteed to be of size [row, col] for generating samples
        # If Z_batch was constructed differently, G_sample might not match X_batch for discriminator
        
        # disc_loss_fake = criterion_gan(fake_output, torch.zeros_like(fake_output).to(device))
        
        disc_loss_fake = criterion_gan(fake_output, fake_labels)
        
        disc_loss = disc_loss_real + disc_loss_fake
        disc_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(discriminator_pt.parameters(), max_norm=1.0)
        
        
        disc_optimizer.step()
        dloss_epoch = disc_loss.item() # Keep last dloss of the steps

    # rrep_dstep, grep_dstep = r_rep.detach(), g_rep_dstep.detach() # From last D step

    gloss_epoch = 0
    # Train Generator
    for _ in range(ng_steps):
        gen_optimizer.zero_grad()
        
        # Generate fake samples (again, use Z_batch_for_g for consistent sizing)
        generated_samples = generator_pt(Z_batch)
        gen_fake_output, _ = discriminator_pt(generated_samples) # We only need logits for G loss
        
        gen_loss = criterion_gan(gen_fake_output, torch.ones_like(gen_fake_output).to(device))
        
        # gen_loss = criterion_gan(gen_fake_output, torch.full_like(gen_fake_output, 0.9).to(device))
        
        
        gen_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(generator_pt.parameters(), max_norm=1.0) # Clip gradients

        gen_optimizer.step()
        gloss_epoch = gen_loss.item() # Keep last gloss of the steps

    # rrep_gstep, grep_gstep = discriminator_pt(X_batch)[1].detach(), discriminator_pt(generator_pt(Z_batch_for_g))[1].detach()

    # scheduler_g.step()
    # scheduler_d.step()

    if i % 100 == 0: # Original TF code printed every iteration
        print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i, dloss_epoch, gloss_epoch)) 






## Data Generation after the training of GAN


feat_size = n_sample
# Size of Generated Data
batch_size_gen_np = (np.arange(0.25, 1.75, 0.25) * feat_size).astype(int)

for i in range(len(batch_size_gen_np)): 
    current_batch_size = batch_size_gen_np[i]
    row1_gen = current_batch_size - Xnew.shape[0]
    if row1_gen < 0: row1_gen = 0

    da1_gen = sample_Z(row1_gen, n_features)
    
    current_xnew_np_gen = Xnew
    # if current_xnew_np_gen.ndim == 1:
    #     current_xnew_np_gen = current_xnew_np_gen.reshape(1, -1)

    Z_batch_gen_np = np.vstack((da1_gen, current_xnew_np_gen))


    Z_batch_gen = torch.tensor(Z_batch_gen_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        g_plot_tensor = generator_pt(Z_batch_gen)
    
    g_plot_np = g_plot_tensor.cpu().numpy()
    
    # Saving the generated Data
    # Ensure directory /vol/eph/data/ exists or change path
    
    file_dir = "/path/for/the/generated/data/generated_data_0_leak"
    
    file_path = file_dir + f"/yan_data_mixdata_iter{i}_top_{k}.csv" 
    

    with open(file_path, "wb") as f:
        g_plot_pd = pd.DataFrame(g_plot_np)
        g_plot_pd.columns = g_plot_pd.columns.astype(str)
        g_plot_pd.to_csv(f)
    print(f"Saved {file_path}, shape: {g_plot_pd.shape}")







