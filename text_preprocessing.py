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
from utils.preprocessing_utils import * 
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--assign_labels_dict', type=str, required=True)
parser.add_argument('--stemming', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

def prepare_data(df, model, stemming):
  '''
    - df    : DataFrame for preparation
    - model : Word vectors for calculating sentence vectors
  '''

  df = df.drop_duplicates().reset_index(drop=True)
  df['text'] = df['text'].str.lower()
  df['text'] = [text.replace('_', ' ') for text in df['text']] # '_' can be used between words

  for k in range(len(df)):
    df['text'][k] = re.sub(r'http\S+', '', str(df['text'][k])) # remove links
    df['text'][k] = re.sub(r'@\S+', '', str(df['text'][k])) # remove usertags
    
  df['text'] = df['text'].apply(lambda text: remove_punctuation(text))


  docs = df['text']
  docs_list = []

  WPT = nltk.WordPunctTokenizer()
  stop_word_list = nltk.corpus.stopwords.words('turkish')
      
  for doc in docs:
      doc = re.sub("\d+"," ",doc) # remove numbers
      doc = WPT.tokenize(doc)
      filtered_tokens = [item for item in doc if item not in stop_word_list]
      lemma = nltk.WordNetLemmatizer()
      lemma_word = [lemma.lemmatize(word) for word in filtered_tokens]

      doc = " ".join(lemma_word)

      docs_list.append(doc)

  df['text'] = pd.DataFrame(docs_list, columns=['text'])

  filter = []
  for i in df['text']: # remove sample if all words are stop words  
    if len(i) == 0:
      filter.append(False)
    else:
      filter.append(True)
  df = df[filter]
  df = df.reset_index(drop=True)

  if stemming:
    stemmer = TurkishStemmer()
    for i, text in enumerate(df['text']):
      word_list = text.split()
      new_sentence = ""
      for word in word_list:
        new_word = stemmer.stem(word)
        new_sentence += (" " + new_word)

      df['text'][i] = new_sentence

  df['vectors'] = [np.zeros(300) for i in range(len(df))]
  not_found = 0
  for i,tweet in enumerate(df['text']):
    bow, count = find_bow([tweet])
    word_list = list(bow.keys()) # finding bag of words for a sample
    avg_vector = np.zeros(300) 
    for word in word_list:
      try:
        avg_vector += model.get_vector(word) # finding vectors of these words
      except:
        not_found += 1
    if count != 0:
      df['vectors'][i] = avg_vector/count # taking average of all words in bag of words for this sample 
  
  print("Average words which are not found in sentences : " + str(not_found/len(df)))

  # drop sample if any word in sample is not in vocabulary
  drop_list = []
  [drop_list.append(k) for k,i in enumerate(df['vectors']) if i.sum() == 0]
  print("Dropped lines after vectorization: " + str(len(drop_list)))
  df = df.drop(drop_list).reset_index(drop=True)
  print("Number of remaining samples: " + str(len(df)))

  print("Value counts for categories :\n")
  print(df['category'].value_counts())

  return df.drop(['vectors'], axis=1), df['vectors']

if __name__ == "__main__":

    file_path = args.file_path
    df = pd.read_csv(file_path)
    
    stemming = args.stemming
    output_dir = args.output_dir

    if stemming == "True":
      stemming_tf = True
    elif stemming == "False":
      stemming_tf = False 
    
    word2vec_file = args.model_path

    model = KeyedVectors.load_word2vec_format(word2vec_file)

    with open(args.assign_labels_dict, 'rb') as f:
      assign_labels_dict = pickle.load(f)

    df, vectors = prepare_data(df, model, stemming_tf)
    df = assign_labels(df, assign_labels_dict)

    df.to_csv(output_dir + "/preprocessed.csv", index=False)
    with open(output_dir + '/vectors.npy', 'wb') as f:
      np.save(f, vectors, allow_pickle=True)