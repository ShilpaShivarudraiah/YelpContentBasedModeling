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


def classify(text: str = ""):
    text = preprocess(text)
    #print("TEXT : ", text, " \n ")
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = ['negative','neutral','positive']
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to('cuda')
    #print("Text",text)
    encoded_input = tokenizer(text,max_length=514, truncation=True, return_tensors='pt').to('cuda')
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
        #print(f"{i + 1}) {l_1} {np.round(float(s), 4)}")
    #print("Label Max",max_label)
    return max_label
      


# TextClassification().classify("good night")
