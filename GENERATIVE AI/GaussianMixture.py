#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:49:21 2024

@author: cornelius
"""


# -----------------------------------------------------------------------------
#                               IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
#                           DATA PREPARATION
# -----------------------------------------------------------------------------

digits = load_digits()

X,y = digits.data, digits.target # Shape of X (#Samples,width_px=8,height_px=8)
                                 # Shape of Y (#Samples,) only the labels

print(f"The shape of X is {X.shape}")
print(f"The shape of y is {y.shape}")

# ------------------- Show random sample as image -----------------------------

image = X[13]
image = np.reshape(image,[8,8])
plt.imshow(image,cmap='gray')
plt.axis('off')


# -----------------------------------------------------------------------------
#                           GAUSSIAN MIXTURE
# -----------------------------------------------------------------------------

n_components = 20

gmm = GaussianMixture(n_components,covariance_type='full',random_state=42)

gmm.fit(X)



# -----------------------------------------------------------------------------
#                            GENERATIVE PART
# -----------------------------------------------------------------------------

n_samples = 100

samples,_ = gmm.sample(n_samples)

samples = np.reshape(samples, [n_samples,8,8])


# -----------------------------------------------------------------------------
#                             VISUALIZATION
# -----------------------------------------------------------------------------


fig, ax = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for k in range(10):
        ax[i][k].imshow(samples[i+k*10],cmap='gray')
        ax[i][k].axis('off')
        
        
        
















