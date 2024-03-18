#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:51:12 2024

@author: cornelius
"""

"""

----------------------------- EXTRACTIVE TEXT SUMMARY ---------------------------

                    1. Tokenize the text -> List of tokens
                    2. Get rid of puntuation and stop words
                    3. Count the number of times a word is used
                    4. Normalize the count with the highest count appearing
                    5. Take the text and for each sentence calculate
                       the normalized count 
                    6. Extract a percentage of the highest ranked sentences:
                       These function as the summary of the text

        


"""

#------------------------------- IMPORTS ---------------------------------------


import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import sent_tokenize


# ------------------------------ TEXT IMPORT -----------------------------------

"""

                    I asked ChatGPT to write a 1000 word text about 
                    the importance of climate science. The result is 
                    stored in the text file imported below.

"""


with open("ClimateScienceText.txt") as f:
    text = f.read()

#print(text)

# ------------------------------ TOKENIZATION ----------------------------------

def tokenize(text):
    
    """
            1. Turn text into lower case letters
            2. Tokenize the text.
            3. Drop punctuation.
            4. Drop stop words
            
    """
    
    text = text.lower()
    punctuation = ["?",".","!",",",";",":","'","-","_"]
    stop_words = list(set(stopwords.words('english')))
    tokenized_text = nltk.word_tokenize(text)
    all_words = [word for word in tokenized_text if word not in punctuation]
    all_words = [word for word in all_words if word not in stop_words]
    return all_words
    



# --------------------------- COUNTING WORDS -----------------------------------




def word_frequencies(wordlist):
    
    """
            1. Count how often each word appears in the text
            2. Determine the highest count
            3. Normalize all counts with the maximum count
    
    """
    word_frequencies = {}

    for word in wordlist:
        if word not in word_frequencies.keys():
            word_frequencies[word]=1
        else:
            word_frequencies[word]+=1
    
    maximum = max(word_frequencies.values())
    
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/maximum
        
    return word_frequencies
    



    


# ----------------------------      SUMMARY    ---------------------------------


def summary(text):
    
    """
            1. Split text into sentences.
            2. Determine the sum of word frequencies for each sentence
            3. Determine a percentage of the sentences with the highest count
               -> this is the summary

    """

    text = text.lower()
    sentences = sent_tokenize(text)
    word_list = tokenize(text)
    word_freq = word_frequencies(word_list)
    
    sentence_count = {}
    for sentence in sentences:
        sentence_count[sentence]=0
    
    for sentence in sentences:
        
        tokens = nltk.word_tokenize(sentence)
        
        for token in tokens:
            if token in word_list:
                sentence_count[sentence]+=word_freq[token]
        
    percentage = 0.2
    
    
    sorted_values = sorted(sentence_count.values()) 
    top_values = [sorted_values[i] for i in range(int(percentage*len(sentences)))]
    
    top_sentences = [sentence for sentence in sentences if sentence_count[sentence] in top_values]
    
    # Turn every first letter of a sentence into a capital letter
    
    final_sentences = []
    
    for sentence in top_sentences:
        
        string = ""
        string += sentence[0].upper()
        string += sentence[1:]
        final_sentences.append(string)
    
    # Return the summary as a string
    
    summary = ""
    for sentence in final_sentences:
        summary += sentence
        summary += " "
    
    
    return summary




print(summary(text))    




















