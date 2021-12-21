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
from utils.kmeans_utils import *
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_path', type=str, required=True)
args = parser.parse_args()

def train_kmeans(df):
  
    X = np.array([np.array([*arr]) for arr in [vector for _,vector in enumerate(df["vectors"], 0)]])
    y = np.array(df['labels'])

    kmeans = KMeans(n_clusters=len(df['labels'].unique()))
    kmeans.fit(X)

    df['clusters'] = kmeans.labels_

    avg_ent = average_entropy(df)

    return df, kmeans, avg_ent


if __name__ == "__main__":

    preprocessed_path = args.preprocessed_path

    df = pd.read_csv(preprocessed_path + '/preprocessed.csv')
    with open(preprocessed_path + '/vectors.npy', 'rb') as f:
        vectors = np.load(f, allow_pickle=True)
    
    df['vectors'] = vectors
    df, kmeans, avg_ent =  train_kmeans(df)

    print("Average entropy: " + str(avg_ent))
    with open(preprocessed_path + "/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    df.to_csv(preprocessed_path + "/df_with_clusters.csv", index=False)