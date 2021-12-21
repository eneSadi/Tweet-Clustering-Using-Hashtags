import pandas as pd
import string
import string
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer

def remove_punctuation(text):
  PUNCT_TO_REMOVE = string.punctuation
  return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def find_bow(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index 
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])

    return bow, len(word_index)

def assign_dict_clusters(assign_dict, kmeans, df):
  '''
    - assign_dict (dict)    : dictionary for labels of hashtags
    - kmeans (model)        : fitted kmeans model for taking its cluster labels
    - df (DataFrame) : used dataframe for training 
  '''
  assign_dict_clusters = {}

  for cluster in np.unique(kmeans.labels_):
    index = df[df['clusters'] == cluster]['labels'].value_counts().reset_index().iloc[0]['index']
    assign_dict_clusters[cluster] = assign_dict[index]
    
  return assign_dict_clusters

def assign_labels(df, assign_label_dict):
  df['labels'] = np.zeros((len(df)), dtype=int)
  for i in range(len(df)):
    df['labels'][i] = assign_label_dict[df['category'][i]]
  df = df.sample(frac=1).reset_index(drop=True)
  return df