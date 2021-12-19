from TurkishStemmer import TurkishStemmer
import pandas as pd
import re
import numpy as np
import string
from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, required=True)
parser.add_argument('--keywords', type=str, required=True)
parser.add_argument('--label', type=int, required=True)
parser.add_argument('--stemming', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

def remove_punctuation(text):
  PUNCT_TO_REMOVE = string.punctuation
  return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def clean_tweets(file_path, keywords, stemming):

  '''
    - file_path (str) : given path for csv file which has tweets included
    - keywords (list) : keyword list which is used while scraping tweets
    - stemming (str)  : stemming operation option, must be True or False
  '''

  df = pd.read_csv(file_path, sep='\t')

  df = df.drop_duplicates().reset_index(drop=True)

  remove_features = ['id', 'conversation_id', 'created_at', 'date', 'time', 'timezone',
       'user_id', 'username', 'name', 'place', 'language',
       'urls', 'photos','mentions',
       'cashtags', 'link', 'retweet', 'quote_url', 'video',
       'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest', 'replies_count', 'retweets_count',	'likes_count']
  df = df.drop(remove_features, axis=1)
  df['tweet'] = df['tweet'].str.lower()
  filter = []
  for i in df['hashtags']:
    if (i.count(',')+1) <= len(keywords):
      filter.append(True)
    else:
      filter.append(False)
  df = df[filter]
  df = df.reset_index(drop=True)

  df['tweet'] = [i.replace(keywords[0] and keywords[1], '') for i in df['tweet']]
  df['tweet'] = [i.replace(keywords[0] or keywords[1], '') for i in df['tweet']]
  for k,i in enumerate(df['tweet']):
    df['tweet'][k] = re.sub(r'http\S+', '', df['tweet'][k])

  for k,i in enumerate(df['tweet']):
    df['tweet'][k] = re.sub(r'@\S+', '', df['tweet'][k])

  filter = []
  for i in df['tweet']:
    if len(i) == 0:
      filter.append(False)
    else:
      filter.append(True)
  df = df[filter]
  df = df.reset_index(drop=True)

  stop_words=pd.read_csv('turkish-stopwords.txt', sep=" ", header=None)
  stop_words.columns=['words_list']

  pat2 = r'\b(?:{})\b'.format('|'.join(list(stop_words['words_list'].str.lower())))
  df['tweet'] = df['tweet'].str.lower().str.replace(pat2, '')

  df['tweet'] = df['tweet'].apply(lambda text: remove_punctuation(text))

  if stemming:
    stemmer = TurkishStemmer()
    for i, tweet in enumerate(df['tweet']):
      word_list = tweet.split()
      new_sentence = ""
      for word in word_list:
        new_word = stemmer.stem(word)
        new_sentence += (" " + new_word)

      df['tweet'][i] = new_sentence

  word_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)
  df['vectors'] = [np.zeros(400) for i in range(len(df))]

  not_found = 0

  for i,tweet in enumerate(df['tweet']):
    word_list = tweet.split()
    avg_vector = np.zeros(400) 
    count = 0
    for word in word_list:
      try:
        avg_vector += word_vectors.word_vec(word)
        count += 1 
      except:
        not_found += 1
    if count != 0:
      df['vectors'][i] = avg_vector/count
  print(str(not_found) + " words are not found")

  drop_list = []
  [drop_list.append(k) for k,i in enumerate(df['vectors']) if i.sum() == 0]
  print("Dropped lines " + str(len(drop_list)))
  df = df.drop(drop_list).reset_index(drop=True)

  print("Number of tweets: " + str(len(df)))

  return df.drop(['vectors'], axis=1), df['vectors']

if __name__ == "__main__":

    file_path = args.file_path
    keywords = args.keywords.split()
    stemming = args.stemming
    output_dir = args.output_dir

    if stemming == "True":
      stemming_tf = True
    elif stemming == "False":
      stemming_tf = False 
    
    df, vectors = clean_tweets(file_path, keywords, stemming_tf)
    df['labels'] = np.zeros((len(df)))
    df['labels'] = args.label

    df.to_csv(output_dir + "/preprocessed_" + keywords[0] + ".csv", index=False)
    with open(output_dir + '/vectors_' + keywords[0] + '.npy', 'wb') as f:
      np.save(f, vectors, allow_pickle=True)