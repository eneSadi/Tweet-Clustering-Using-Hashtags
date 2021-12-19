from gensim.utils import dict_from_corpus
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessed_path', type=str, required=True)
args = parser.parse_args()

def assign_dict_clusters(assign_dict, kmeans, merged_df):
  assign_dict_clusters = {}

  for cluster in np.unique(kmeans.labels_):
    index = merged_df[merged_df['clusters'] == cluster]['labels'].value_counts().reset_index().iloc[0]['index']
    assign_dict_clusters[cluster] = assign_dict[index]
    
  return assign_dict_clusters


def KMeans_Model(preprocessed_path):
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
            merged_df = pd.concat([merged_df, df_tmp])
    
    merged_df.reset_index(drop=True)
    merged_df.sample(frac=1).reset_index(drop=True)

    X = np.array([np.array([*arr]) for arr in [vector for _,vector in enumerate(merged_df["vectors"], 0)]])
    y = np.array(merged_df['labels'])

    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(X)

    merged_df['clusters'] = kmeans.labels_

    dict_for_clusters = assign_dict_clusters(assign_dict, kmeans, merged_df)

    for i in range(n_cluster):
        tmp_df = pd.DataFrame(merged_df[merged_df['clusters'] == i]['labels'].value_counts()).reset_index()
        for k in range(len(tmp_df)):
            tmp_df['index'][k] = assign_dict[tmp_df['index'][k]]
        print("\n\n\nFrequencies for cluster " + str(i) + " :")
        print(tmp_df)
        print("\n\n\n")
    return kmeans, dict_for_clusters


if __name__ == "__main__":

    preprocessed_path = args.preprocessed_path
    kmeans, dict_for_clusters =  KMeans_Model(preprocessed_path)

    with open(preprocessed_path + "/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open(preprocessed_path + "/dict_for_clusters.pkl", "wb") as f:
        pickle.dump(dict_for_clusters, f)