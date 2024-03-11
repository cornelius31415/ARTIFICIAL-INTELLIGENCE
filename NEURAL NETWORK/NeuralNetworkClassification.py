#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:36:12 2024

@author: cornelius
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

df = pd.read_csv("mnist_data.csv")
print(df)
print(df.info())
print()
data = df.values.tolist()
#print(data[:2])

label = df.label.values.tolist()
features = df.drop("label",axis=1).values.tolist()


#liste = [0,255,0,125,45,255,67,0,0]
#image = np.array(liste).reshape((3,3))
#plt.imshow(image,cmap="Greys")

plot = data[-2]
image = np.asfarray(plot[1:]).reshape(28,28)
plt.imshow(image,cmap="Greys")



input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 1e-3

features = (np.asfarray(features)/255*0.99)+0.01

zahl1 = features[0]


w_ih = np.random.rand(hidden_nodes,input_nodes)-0.5
w_ho = np.random.rand(output_nodes,hidden_nodes)-0.5

# feedforward(): Signalübertragung
# training(): Signalübertragung, Fehler berechnet, Gewichte aktualisiert


def feedforward(input_list,w_ih,w_ho):
    
    input_list = np.array(input_list,ndmin=2)
    input_hidden = input_list @ w_ih.T
    output_hidden = scipy.special.expit(input_hidden)
    input_final = output_hidden @ w_ho.T
    output_final = scipy.special.expit(input_final)
    
    
    
    return output_final


print(feedforward(zahl1,w_ih,w_ho))



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
    
    

for i in range(0,int(0.8*len(features))):
    target_vector = np.zeros(output_nodes)
    target_vector[int(label[i])] = 0.99
    train(features[i],w_ih,w_ho,target_vector,learning_rate)
    
    if (i%1000)==0:
        print("Training")
    


print(feedforward(features[-2], w_ih, w_ho))




































