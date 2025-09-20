## wasserstein distance

import pandas as pd

import ot 
import numpy as np
from scipy.spatial.distance import cdist


## loading the real data

## Muraro files don't need the index_col = 0 but need header = 0 (both df1, df2)
## CBMC needs index_col = 0 and header = 0 (both df1, df2)
## pollen needs header = None (both df1, df2)
## yan needs header = None (both df1, df2)

df2 = pd.read_csv(r"data/cell_types/yan_celltype.csv", header = None) 
df1 = pd.read_csv("data/expression_matrix", header=None)


## "YAN" & "CBMC" needs ==> transpose, "POLLEN" & "MURARO" does not need transpose


df1 = df1.T


df1.reset_index(drop=True, inplace=True)



X = df1.values  # Features


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

## generated data loading and WD calculation
data = X
# data=np.transpose(data)
# Enlarged Dataset from GAN
resgan= pd.read_csv('/path/for/the/generated/data/gan_yan_generated_mixdata_iter5.csv',  header = None)
unifs1 = data / len(data)
unifs2 = resgan / len(resgan)
dist_mat = cdist(unifs1, unifs2, 'euclid')
emd_dists = ot.emd2(np.ones(len(data)) / len(data), np.ones(len(resgan)) / len(resgan), dist_mat,numItermax=100000)
print(emd_dists)