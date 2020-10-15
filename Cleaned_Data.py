#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:37:18 2020

@author: burlios
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
import nltk
import spacy

from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


sms = pd.read_csv("SMSSpamCollection.txt", sep = "\t")
# Set columns of interest
sms = sms[["Type", "SMS"]]

####################### Data Cleaning ###################################

#Check for missing data
sms.isnull().any()

# Convert the SMS to string type
sms["SMS"] = sms["SMS"].astype(str)


# Clean the data for processing
# remove all capitalization, punctuation, 

# lowercase 
sms["SMS"] = sms["SMS"].str.lower()

# remove punctuation from each word *****
import string
table = str.maketrans('', '', string.punctuation)
sms['SMS'] = [w.translate(table) for w in sms['SMS']]

# filter out non-alphabetic words
x=[] # create a dummy list to deposit filtered strings 
for k in range(len(sms['SMS'])):
    #print(sms['SMS'][k])
    x.append(''.join([i for i in sms['SMS'][k] if not i.isdigit()])) # .append tacks on strings to x one at a time 
    
sms['SMS'] = x # put x into the column for SMS


# Tokenize the SMSs ****
tokenizer = RegexpTokenizer(r'\w+')
sms['SMS'] = sms['SMS'].apply(lambda x: tokenizer.tokenize(x.lower()))


# remove stopwords ****
stop_words = set(stopwords.words('english'))
x =[]
for k in range(len(sms['SMS'])):
    x.append(' '.join([w for w in sms['SMS'][k] if not w in stop_words]))
sms['SMS'] = x

sms.to_csv('sms_clean.csv')

########################### Lemmatized Words Version #############################
# run this last and before converting csv 

# Lemmatize words 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

porter=PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

# Loop to lemmatize
x=[]
for k in range(len(sms['SMS'])):
    x.append(stemSentence(sms['SMS'][k]))
sms['SMS'] = x

sms.to_csv('sms_clean2.csv')






























