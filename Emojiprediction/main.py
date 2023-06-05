import pickle
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

with open("Emojiprediction/model.pkl","rb") as file:
    model = pickle.load(file)

#nltk.download(stopwords)
stop_words = set(stopwords.words("english"))

sen = st.text_input("Enter your thought: ")
#sen = input("Enter your thought: ")

if sen:
    tokens = nltk.word_tokenize(sen)
    filteredtokens = [token for token in tokens if token.lower() not in stop_words]
    input = ' '.join(filteredtokens)

    input = [input]

    v = TfidfVectorizer()
    v.fit(input)
    x = v.transform(input)

    x = x.toarray()
    desired_features = 46671

# Remove the extra features if the number of features is greater than the desired number
    if x.shape[1] > desired_features:
        x_final = x[:, :desired_features]
    else:
    # Pad with zeros if the number of features is less than the desired number
        extra_features = desired_features - x.shape[1]
        x_final = np.pad(x, ((0, 0), (0, extra_features)), mode='constant')

    output = model.predict(x_final)
    df = pd.read_csv("E:\Projects\Emojiprediction/Mapping.csv")
    map = dict(zip(df["number"],df["emoticons"]))

    st.write(map[output[0]])
#print(map[output[0]])
