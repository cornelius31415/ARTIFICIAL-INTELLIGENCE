#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:31:08 2024

@author: cornelius
"""

# ------------------------ IMPORTS -------------------------------------------

import pandas as pd
import numpy as np
import scipy.special
from scipy.special import softmax
import random
from sklearn.preprocessing import LabelEncoder
import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer





# ------------------------ PREPARING THE DATA ---------------------------------




df = pd.read_csv("ChatbotTraining.csv")
label_encoder = LabelEncoder()
df["tag"] = label_encoder.fit_transform(df["tag"])
patterns = df["patterns"].values.tolist()
tags = df["tag"].values.tolist()
tag_amount = len(set(df["tag"].values.tolist()))
responses = df["responses"].values.tolist()
responses = list(zip(tags,responses))





# ------------------------ HOW THE CHATBOT WILL WORK --------------------------

"""
                I.  TRAINING
                
1. Take a list of sentences with tags 
2. Tokenize the sentences
3. Stem the words in the sentences
4. Turn the sentences into bags of words
5. Each bag of word goes with a label
6. All bags of words together is the training data


                II. CHATTING

1. Get a sentence
2. Tokenize the sentence
3. Stem the sentence
4. Make a bag of words
5. Classifiy bag of words
6. choose random from response list with corresponding tag
 


"""



# ------------------------ TOKENIZATION --------------------------------------

                # Sentence gets splitted into array of words/tokens
def tokenize(sentence):
    return nltk.word_tokenize(sentence)



# ------------------------ STEMMING ------------------------------------------

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
            bag[i] = 1
    return bag
        



# ----------------------- CREATE A LIST OF ALL WORDS---------------------------

ignore_words = ["?",".","!"]
words = []

for sentence in patterns:
    
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence if word not in ignore_words]
    words.extend(sentence_words)
    
for sentence in df["responses"].values.tolist():
        
    sentence = tokenize(sentence)
    sentence_words = [stem(word) for word in sentence if word not in ignore_words]
    words.extend(sentence_words)



# ------------------------- NEURAL NETWORK -----------------------------------



input_nodes = len(words)
hidden_nodes = 1000
output_nodes = tag_amount
learning_rate = 1e-1
epochs = 15





w_ih = np.random.rand(hidden_nodes,input_nodes)-0.5
w_ho = np.random.rand(output_nodes,hidden_nodes)-0.5




def feedforward(input_list,w_ih,w_ho):
    
    input_list = np.array(input_list,ndmin=2)
    input_hidden = input_list @ w_ih.T
    output_hidden = scipy.special.expit(input_hidden)
    input_final = output_hidden @ w_ho.T
    output_final = scipy.special.expit(input_final)
    return output_final






def train(input_list,w_ih,w_ho, target_list,learning_rate):
    target_list = np.array(target_list,ndmin=2) # Neu
    input_list = np.array(input_list,ndmin=2)
    input_hidden = input_list @ w_ih.T
    output_hidden = scipy.special.expit(input_hidden)
    input_final = output_hidden @ w_ho.T
    output_final = scipy.special.expit(input_final)
    
    output_errors = target_list - output_final
    hidden_errors = output_errors @ w_ho
    
    w_ho += learning_rate*np.dot((output_errors*output_final*(1-output_final)).T,output_hidden)
    w_ih += learning_rate*np.dot((hidden_errors*output_hidden*(1-output_hidden)).T,input_list)
    
  
    
  
# ------------------------------- TRAINING ------------------------------ ---- 
print()
print("     I am sorry I have to prepare first.")

print()
  
for e in range(epochs):
    for i in range(len(patterns)):
        target_vector = np.zeros(output_nodes)
        target_vector[int(tags[i])] = 0.99
        train(bag_of_words(patterns[i],words),w_ih,w_ho,target_vector,learning_rate)
    
        if (i%10000)==0:
            print("                  PREPARING                  ")
            print()






# --------------------------- CHATTING ---------------------------------------


# THIS FUNCTION GIVES A LABEL FOR A SENTENCE WRITTEN BY A USER
def classify(sentence):
    sentence = bag_of_words(sentence, words)
    neural_output = feedforward(sentence, w_ih, w_ho)
    neural_output = softmax(neural_output)
    neural_output = neural_output.tolist()
    neural_output = max(neural_output)
    label = neural_output.index(max(neural_output))
    return label
    
    

# HERE THE ACTUAL COMMUNICATION STARTS

while True:

    user_input = input("Your Message: ")
    print()

    response_options = [tupel[1] for tupel in responses if tupel[0]==classify(user_input)]
    print(random.choice(response_options))
    print()









