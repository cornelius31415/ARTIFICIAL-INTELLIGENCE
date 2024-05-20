#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:04:52 2024

@author: cornelius
"""

"""
                Downloading the 8x8 px MNIST data from the 
                sklearn library to turn them into a csv file
                and save it locally.
                

"""


# -----------------------------------------------------------------------------
#                               IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import load_digits
import pandas as pd

# -----------------------------------------------------------------------------
#                           DATA LOADING
# -----------------------------------------------------------------------------

digits = load_digits()

features, labels = digits.data, digits.target   # Shape of X (#Samples,width_px=8,height_px=8)
                                                 # Shape of Y (#Samples,) only the labels

print(f"The shape of X is {features.shape}")
print(f"The shape of y is {labels.shape}")

mnist = pd.DataFrame(features)
mnist['labels'] = labels

print(mnist)


mnist.to_csv("MNIST_8x8px_images",index=False)      # index=False such that no index column is added




