# Wasserstein Distance

## Overview

Wasserstein distance, or Earth Mover's Distance, measures the minimum "cost" to transform one probability distribution into another. This metric assesses the difference between probability distributions
within a given metric space. In data generation model like ours, we have to decide the level of similarity between different the real data and the generated data. In order to do this, we need to find a function or operation for generating a score which “measures” the level of similarity. This function can take many forms, but one common popular metric for task like this is the Wasserstein distance (W D). Basically the WD is the measurement of the similarity between the real and generated data distribution. 

### Advantages

- Efficient for comparing distributions that may not align perfectly.
- Captures spatial relationships in distributions.

### Limitations

- Provides an approximation to similarity and may not guarantee exact results.

### Use Cases

- Compare probability distributions in statistics.
- Apply in image processing and machine learning.

## WD in GAT-GAN:

``` python
import ot 
import numpy as np
from scipy.spatial.distance import cdist
# Original preprocessed dataset (default yan dataset)
data = df.to_numpy()
# data=np.transpose(data)
# Enlarged Dataset from GAN
resgan=pd.read_feather('/vol/eph/data/cbmc_dataset/data_cbmc_iter4.feather') #np.genfromtxt('/vol/eph/data/data_mixdata_iter4.csv',delimiter=",") 
unifs1 = data / len(data)
unifs2 = resgan / len(resgan)
dist_mat = cdist(unifs1, unifs2, 'euclid')
emd_dists = ot.emd2(np.ones(len(data)) / len(data), np.ones(len(resgan)) / len(resgan), dist_mat,numItermax=100000)
print(emd_dists)
```

After generating the sythetic cell samples using GAN, the WD comes into the charge of data validation. Basically through WD the similarities between different generated data and real data is measured. If the certain distance is too small(<<0.1) then we can say the data distribution of the generateddata to the real data is quite similer to the real data. As an example we can say, the CBMC dataset exhibits an
exceptionally low Wasserstein Distance of 0.00324, demonstrating an almost perfect match between the generated and real data. This suggests that the feature selection method is extremely effective for this dataset.
Also this same generated data distribution and data similaries can be identified using differnt parameters, as: 
- By identifying real priority nodes
- By run the GAN models with more and more epochs to reduce generator loss.
- identify the best possible generated dataset to use further.
