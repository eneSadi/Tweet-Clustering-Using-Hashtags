import pandas as pd
import string
from gensim.models import KeyedVectors
import re
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_path', type=str, required=True)
args = parser.parse_args()

def remove_punctuation(text):
  PUNCT_TO_REMOVE = string.punctuation
  return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def predict_tweet(text, dict_for_clusters, kmeans):
  stop_words=pd.read_csv('turkish-stopwords.txt', sep=" ", header=None)
  stop_words.columns=['words_list']

  pat2 = r'\b(?:{})\b'.format('|'.join(list(stop_words['words_list'].str.lower())))
  text = text.lower().replace(pat2, '')

  text = remove_punctuation(text)

  word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)

  not_found = 0

  word_list = text.split()
  avg_vector = np.zeros(400) 
  count = 0
  for word in word_list:
    try:
      avg_vector += word_vectors.word_vec(word)
      count += 1 
    except:
      not_found += 1

  text_vector = avg_vector/count
  text_vector = text_vector.reshape(1, -1)
  
  return dict_for_clusters[kmeans.predict(text_vector)[0]]


if __name__ == "__main__":

    with open(args.preprocessed_path + "/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(args.preprocessed_path + "/dict_for_clusters.pkl", "rb") as f:
        dict_for_clusters = pickle.load(f)

    text = input("Enter your tweet text for prediction: ")
    prediction = predict_tweet(text, dict_for_clusters, kmeans)

    print("\nThis tweet is about to \n" + prediction + "\n\n")