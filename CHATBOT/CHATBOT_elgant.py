#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:08:13 2024

@author: cornelius
"""

"""
        I wanted to implement a simple Chatbot in as little code
        as possible. Instead of working with neural networks I 
        chose decision trees. 

"""


# ------------------------         IMPORTS         ---------------------------


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("ChatbotTraining.csv")


# ------------------------ BAG OF WORDS DATAFRAME  ---------------------------


sentences = df['patterns'].values.tolist()


count_vec = CountVectorizer()
word_counts = count_vec.fit_transform(sentences)
bag_of_words_df = pd.DataFrame(word_counts.toarray(),columns = count_vec.get_feature_names_out())



# ---------------------------    DECISION TREE    ----------------------------



decision_tree = DecisionTreeClassifier()
decision_tree.fit(bag_of_words_df.values, df['tag'].values)



# -----------------------------    CHATBOT     -------------------------------


print()
print()
print("---------------------------------------------------------------")
print("----------------     WELCOME TO CHATTY      -------------------")
print("---------------------------------------------------------------")
print("\n")

while True:

    user_input = input("Your Message: ")
    print()
    
    prediction = decision_tree.predict(count_vec.transform([user_input]))[0]
    response_options = df.loc[df['tag']==prediction,'responses'].values.tolist()
    print(random.choice(response_options))
    
    print()




















