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
    
    X=data
    y=label
    
    # Create a KNN graph for the adjacency matrix with 5 nearest neighbors
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)
    adjacency_matrix = torch.zeros((len(X), len(X)), dtype=torch.float32)

    for i in range(len(X)):
        adjacency_matrix[i, indices[i]] = 1.0

    # Convert adjacency matrix to a COO format tensor
    row, col = adjacency_matrix.nonzero().t()
    edge_index = torch.stack([row, col], dim=0)


    priority_nodes = torch.tensor(index_list, dtype=torch.long)

    # Create a PyTorch data object
    # Here the values x, edge_index, y and priority nodes are initialized together
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        priority_nodes = priority_nodes
    )
    
    data = data.to(device)

    

    class GATClassifier(nn.Module):
        def __init__(self, num_features, num_classes, priority_weight):
            super(GATClassifier, self).__init__()
            self.conv1 = GATConv(num_features, 32, heads=8)
            self.conv2 = GATConv(32 * 8, num_classes, heads=1)
            self.priority_weight = priority_weight

        def forward(self, data):
            x, edge_index, priority_nodes = data.x, data.edge_index, data.priority_nodes

            # First GAT layer
            x, attention1 = self.conv1(x, edge_index, return_attention_weights=True)

            # Modify attention scores to give more priority to specific nodes
            attention = torch.ones(x.size(0), device=x.device)  # Initialize with ones
            attention[priority_nodes] += self.priority_weight  # Increase attention to priority nodes
            x = x * attention.view(-1, 1)  # Element-wise multiplication with attention scores

            x = torch.relu(x)

            # Second GAT layer
            x, attention2 = self.conv2(x, edge_index, return_attention_weights=True)

            # You can now access attention coefficients for each node from attention1 and attention2

            return x, attention1, attention2


    priority_weight=2
    num_features = X.shape[1]        # Number of features
    num_classes = len(np.unique(y))  # Number of unique class labels
    model = GATClassifier(num_features, num_classes, priority_weight)      # Model Compile
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 7501
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output, attention1, attention2 = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')





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

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# ### Generator


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hsize=[16, 16]):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc_out = nn.Linear(hsize[1], output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2) # Common default for LeakyReLU

    def forward(self, z):
        h1 = self.leaky_relu(self.fc1(z))
        h2 = self.leaky_relu(self.fc2(h1))
        out = self.fc_out(h2) # No activation on output, matches TF dense layer default
        return out


########## NEWER ##########################

# class Generator(nn.Module):
#     def __init__(self, input_dim, output_dim, hsize=[512, 512]): # Increased capacity
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hsize[0])
#         self.bn1 = nn.BatchNorm1d(hsize[0]) # BatchNorm
#         self.fc2 = nn.Linear(hsize[0], hsize[1])
#         self.bn2 = nn.BatchNorm1d(hsize[1]) # BatchNorm
#         self.fc_out = nn.Linear(hsize[1], output_dim)
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         # Consider Tanh if your x_plot data is scaled to [-1, 1]
#         # self.tanh = nn.Tanh()

#     def forward(self, z):
#         h1 = self.leaky_relu(self.bn1(self.fc1(z)))
#         h2 = self.leaky_relu(self.bn2(self.fc2(h1)))
#         out = self.fc_out(h2) # No activation for general real-valued output
#         # if using Tanh: out = self.tanh(self.fc_out(h2))
#         return out


# ### Discriminator


class Discriminator(nn.Module):
    def __init__(self, input_dim, hsize=[16, 16]):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hsize[0])
        self.fc2 = nn.Linear(hsize[0], hsize[1])
        self.fc3 = nn.Linear(hsize[1], input_dim) # h3 layer, outputting 'col' features
        self.fc_out = nn.Linear(input_dim, 1) # Logits
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x_in):
        h1 = self.leaky_relu(self.fc1(x_in))
        h2 = self.leaky_relu(self.fc2(h1))
        h3 = self.fc3(h2) 
        out_logits = self.fc_out(h3)
        return out_logits, h3


############ NEWER ##############################

# class Discriminator(nn.Module):
#     def __init__(self, input_dim, hsize=[512, 256]): # Asymmetric capacity, D can be shallower
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hsize[0])
#         # No BatchNorm on the first layer of D is a common practice, but can be added
#         self.fc2 = nn.Linear(hsize[0], hsize[1])
#         self.bn2 = nn.BatchNorm1d(hsize[1]) # BatchNorm
#         # Simplified output part:
#         self.fc_out = nn.Linear(hsize[1], 1) # Directly map from last hidden layer to 1 logit
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.dropout = nn.Dropout(0.4) # Add dropout to D to prevent overpowering G

#     def forward(self, x_in):
#         h1 = self.leaky_relu(self.fc1(x_in))
#         h1 = self.dropout(h1)
#         h2 = self.leaky_relu(self.bn2(self.fc2(h1))) # Apply BN before activation
#         h2 = self.dropout(h2)
#         # The original h3 layer is removed for simplification.
#         # The role of h3 (intermediate representation) is now implicitly handled by h2.
#         out_logits = self.fc_out(h2)
#         # If you need an intermediate representation similar to h3 for other purposes:
#         # intermediate_rep = h2.detach() # or some transformation of h2
#         # For now, we only need logits for GAN loss.
#         return out_logits 


# Initialize models and optimizers
col = Xnew.shape[1] 
hsize_G = [512, 512] 
hsize_D = [256, 128]

generator_pt = Generator(input_dim=col, output_dim=col, hsize=hsize_G).to(device)
discriminator_pt = Discriminator(input_dim=col, hsize=hsize_D).to(device)


# Loss Function

criterion_gan = nn.BCEWithLogitsLoss()


# Optimizers

lr_g = 0.0002
lr_d = 0.0004

gen_optimizer = optim.RMSprop(generator_pt.parameters(), lr=lr_g)
disc_optimizer = optim.RMSprop(discriminator_pt.parameters(), lr=lr_d)

# gen_optimizer = optim.Adam(generator_pt.parameters(), lr=lr_g, betas=(0.5, 0.999))
# disc_optimizer = optim.Adam(discriminator_pt.parameters(), lr=lr_d, betas=(0.5, 0.999))


# scheduler_g = optim.lr_scheduler.StepLR(gen_optimizer, step_size=5000, gamma=0.7)
# scheduler_d = optim.lr_scheduler.StepLR(disc_optimizer, step_size=5000, gamma=0.7)


# # Instantiate GAN models
# generator = Generator(input_dim, output_dim).to(device)
# discriminator = Discriminator(input_dim).to(device)
# gen_optimizer = optim.Adam(generator.parameters(), lr=lr_gan, betas=(0.5, 0.999))
# disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr_gan, betas=(0.5, 0.999))


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


# ## Wassestine Distance


import ot 
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd



## Instructions for data loading 
## For 0% leakage
## For yan data =>
## For pollen data => generated data should load (inside pandas.read_csv) with header = None
## For cbmc data => generated data should load (inside pandas.read_csv) with header = None
## For muraro data => generated data should load (inside pandas.read_csv) with header = None


## For 10/20/30 % leakage

## For yan data =>
## For pollen data => generated data should load (inside pandas.read_csv) with header = 0, index_col = 0
## For cbmc data => generated data should load (inside pandas.read_csv) with header = 0, index_col = 0
## For muraro data => generated data should load (inside pandas.read_csv) with header = 0, index_col = 0



data = X
# data=np.transpose(data)
# Enlarged Dataset from GAN
resgan= pd.read_csv('/home/bernadettem/bernadettenotebook/Ritwik/NLP/GAT_GAN/generated_data/gan/gan_yan_generated_mixdata_iter5.csv',  header = None)
unifs1 = data / len(data)
unifs2 = resgan / len(resgan)
dist_mat = cdist(unifs1, unifs2, 'euclid')
emd_dists = ot.emd2(np.ones(len(data)) / len(data), np.ones(len(resgan)) / len(resgan), dist_mat,numItermax=100000)
print(emd_dists)


# In[ ]:




