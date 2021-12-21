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

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_path', type=str, required=True)
args = parser.parse_args()

def train_kmeans(preprocessed_path):
    merged_df = pd.DataFrame()
    assign_dict = {}
    n_cluster = 0
    for file in os.listdir(preprocessed_path):
        if "preprocessed" in file:
            n_cluster += 1
            df_tmp = pd.read_csv(preprocessed_path + '/' + file)
            assign_dict[df_tmp['labels'][0]] = file.split('.')[0].split('_')[1]
            vector_name = "vectors_" + file.split('.')[0].split('_')[1] + ".npy"
            if vector_name in os.listdir(preprocessed_path):
                with open(preprocessed_path + '/' + vector_name, 'rb') as f:
                    vectors_tmp = np.load(f, allow_pickle=True)
            else:
                print("Files are not complete")
                exit()

            df_tmp['vectors'] = vectors_tmp
            df = pd.concat([merged_df, df_tmp])
    
    X = np.array([np.array([*arr]) for arr in [vector for _,vector in enumerate(df["vectors"], 0)]])
    y = np.array(df['labels'])

    kmeans = KMeans(n_clusters=len(df['labels'].unique()))
    kmeans.fit(X)

    df['clusters'] = kmeans.labels_

    avg_ent = average_entropy(df)

    return df, kmeans, avg_ent


if __name__ == "__main__":

    preprocessed_path = args.preprocessed_path
    df, kmeans, avg_ent =  KMeans_Model(preprocessed_path)

    print("Average entropy: " + str(avg_ent))
    with open(preprocessed_path + "/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(preprocessed_path + "/df_with_clusters.pkl", "wb") as f:
        pickle.dump(df, f)