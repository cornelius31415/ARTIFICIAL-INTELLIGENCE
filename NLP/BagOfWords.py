#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:44:22 2024

@author: cornelius
"""

"""
                BAG OF WORDS TEMPLATE
        
             to  use  for  NLP  projects

"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

documents = ["this is a sample","hello my dear friend, you are my best friend"]

count_vec = CountVectorizer()
word_counts = count_vec.fit_transform(documents)
df_word_counts = pd.DataFrame(word_counts.toarray(), 
                              columns=count_vec.get_feature_names_out())
print(df_word_counts)