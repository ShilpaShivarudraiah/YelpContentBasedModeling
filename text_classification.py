from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')

"""This is a function named "preprocess" which takes a string input "text" and 
    performs several text preprocessing tasks on it. 
    These tasks include removing punctuation, converting text to lowercase, and removing stop words. 
    The function returns the preprocessed text as a string. 
    If the input text is not a string, it is returned as is."""


def preprocess(text):
    if isinstance(text, str):

        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]

        # Join tokens back into a string
        text = ' '.join(tokens)

        return text
    else:
        return text
    
"""This is a function named classify which takes a string as an input and returns a sentiment classification label for that text. 
    It first preprocesses the input by removing punctuations, converting it to lowercase, and removing stop words. 
    It then uses a pre-trained transformer model called "cardiffnlp/twitter-roberta-base" 
    to classify the sentiment of the input into one of three categories: negative, neutral, or positive. 
    The output is the sentiment label with the highest score."""

def classify(text: str = ""):
    text = preprocess(text)
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = ['negative','neutral','positive']
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to('cpu')
    encoded_input = tokenizer(text,max_length=500, truncation=True, return_tensors='pt').to('cpu')
    output = model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    max_score = 0
    for i in range(scores.shape[0]):
        l_1 = labels[ranking[i]]
        s = scores[ranking[i]]
        if s > max_score:
          max_score = s
          max_label = l_1
    return max_label
      


# TextClassification().classify("good night")
