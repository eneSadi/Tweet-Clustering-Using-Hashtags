from TurkishStemmer import TurkishStemmer
import pandas as pd
import re
import numpy as np
import string
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import string
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
import scipy.stats
import pickle
import argparse
from utils.kmeans_utils import *
from utils.preprocessing_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

def predict_text(text, assign_dict, kmeans, model, df):
  '''
    - text (str)               : string for prediction
    - dict_for_clusters (dict) : dictionary for cluster labels of hashtags
    - kmeans                   : model for predict the cluster of given vector
    - model                    : model for taking vector values of words
  '''
  dict_for_clusters = assign_dict_clusters(assign_dict, kmeans, df)
  
  text = text.lower()
  text = remove_punctuation(text)

  WPT = nltk.WordPunctTokenizer()
  stop_word_list = nltk.corpus.stopwords.words('turkish')

  text = re.sub("\d+"," ",text) # remove numbers
  text = WPT.tokenize(text)
  filtered_tokens = [item for item in text if item not in stop_word_list]
  lemma = nltk.WordNetLemmatizer()
  lemma_word = [lemma.lemmatize(word) for word in filtered_tokens]

  text = " ".join(lemma_word)

  not_found = 0
  bow, count = find_bow([text])
  word_list = list(bow.keys()) # finding bag of words for a sample
  avg_vector = np.zeros(300) 
  for word in word_list:
    try:
      avg_vector += model.get_vector(word) # finding vectors of these words
    except:
      not_found += 1
  if count != 0:
    avg_vector = avg_vector/count # taking average of all words in bag of words for this sample 

  avg_vector = avg_vector.reshape(1,-1)
  prediction = dict_for_clusters[kmeans.predict(avg_vector)[0]]

  return prediction


if __name__ == "__main__":

    with open(args.preprocessed_path + "/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(args.preprocessed_path + "/assign_dict.pkl", "rb") as f:
        assign_dict = pickle.load(f)
    
    df = pd.read_csv(args.preprocessed_path + '/df_with_clusters.csv')

    word2vec_file = args.model_path
    model = KeyedVectors.load_word2vec_format(word2vec_file)

    text = input("\n\nEnter your tweet text for prediction: ")
    prediction = predict_text(text, assign_dict, kmeans, model, df)

    print("\nThis tweet is about to \n" + prediction + "\n\n")






