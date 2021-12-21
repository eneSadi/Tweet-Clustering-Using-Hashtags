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

def ent(data):
    """
    Calculates entropy of the passed `pd.Series`
    """
    p_data = data.value_counts()           # counts occurrence of each value
    entropy = scipy.stats.entropy(p_data)  # get entropy from counts
    return entropy

def average_entropy(df):
    sum_ent = 0
    n_clusters = len(df['clusters'].value_counts())
    for i in range(n_clusters):
    sum_ent += ent(df[df['clusters'] == i]['labels'])
    return sum_ent/n_clusters

def check_cluster_dist(df, cluster):
    ctrl_df = pd.DataFrame(df[df['clusters'] == cluster]['labels'].value_counts()).reset_index()

    for i in range(len(ctrl_df)):
    ctrl_df['index'][i] = assign_dict[ctrl_df['index'][i]]

    return ctrl_df