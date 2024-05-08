#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:31:21 2024

@author: cornelius
"""

# ---------------------------- IMPORTS ---------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import random


# ------------------------ PREPARING THE DATA ---------------------------------


df = pd.read_csv("ChatbotTraining.csv")
label_encoder = LabelEncoder()
df["tag"] = label_encoder.fit_transform(df["tag"])

patterns = df['patterns'].values.tolist()
responses = df["responses"].values.tolist()
tags = df["tag"].values.tolist()
responses = list(zip(tags,responses))
    
labels   = df['tag']


# ----------------------------------------------------------------------------
# ------------------- NATURAL LANGUAGE PROCESSING ----------------------------
# ----------------------------------------------------------------------------



# ------------------------ TOKENIZATION --------------------------------------

        # Sentence gets splitted into array of words/tokens
                
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# ---------------------------- STEMMING --------------------------------------

                       # root of each word
            
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())



# ------------------------ BAG OF WORDS --------------------------------------

                   # return bag of words array

def bag_of_words(sentence,words):
    
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence]
    bag = np.zeros(len(words),dtype=np.float32)
    for i in range(len(words)):
        if words[i] in sentence_words:
            bag[i] += 1
    return bag
        


# ----------------------- CREATE A VOCABULARY --------------------------------


ignore_words = ["?",".","!"]
words = []

for sentence in df["patterns"].values.tolist():
    
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence if word not in ignore_words]
    words.extend(sentence_words)
    
for sentence in df["responses"].values.tolist():
        
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence if word not in ignore_words]
    words.extend(sentence_words)
    


# ----------------------------------------------------------------------------
# ------------------------------ DECISION TREE -------------------------------
# ----------------------------------------------------------------------------

bow_patterns = [bag_of_words(pattern, words) for pattern in patterns]

decision_tree = DecisionTreeClassifier()
decision_tree.fit(bow_patterns, labels)



# HERE THE ACTUAL COMMUNICATION STARTS

while True:

    user_input = input("Your Message: ")
    print()
    sentence = [bag_of_words(user_input, words)]
    prediction = decision_tree.predict(sentence)
    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))
    print()































