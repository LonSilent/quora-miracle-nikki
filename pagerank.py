# coding: utf-8
# Based on notebook by https://www.kaggle.com/shubh24 
# https://www.kaggle.com/shubh24/pagerank-on-quora-a-basic-implementation
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import hashlib

df_train = pd.read_csv('./data/train.csv').fillna("")
df_test = pd.read_csv('./data/test.csv').fillna("")


# Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):
    hash_key1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
    hash_key2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)

def pagerank():
    MAX_ITER = 25
    d = 0.85

    # Initializing -- every node gets a uniform value!
    pagerank_dict = {x: 1 / len(qid_graph) for x in qid_graph}
    num_nodes = len(pagerank_dict)

    for i in range(MAX_ITER):
        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])
            # pagerank = (1-d) / N + (d * sum(pageranks/links)
            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict

def get_pagerank_value(row):
    try:
        q1 = hashlib.md5(row["question1"].encode('utf-8')).hexdigest()
        q2 = hashlib.md5(row["question2"].encode('utf-8')).hexdigest()
    except:
        print(hashlib.md5(row["question1"].encode('utf-8')).hexdigest())
        print(hashlib.md5(row["question2"].encode('utf-8')).hexdigest())
    s = pd.Series({
        "q1_pr": pagerank_dict[q1],
        "q2_pr": pagerank_dict[q2]
    })

    return s

qid_graph = {}
print('Apply to train...')
df_train.apply(generate_qid_graph_table, axis=1)
print('Apply to test...')
df_test.apply(generate_qid_graph_table, axis=1)

print('Main PR generator...')
pagerank_dict = pagerank()

print('Apply to train...')
pagerank_feats_train = df_train.apply(get_pagerank_value, axis=1)
print('Writing train...')
pagerank_feats_train.to_csv("./data/train_pagerank.csv", index=False)
print('Apply to test...')
pagerank_feats_test = df_test.apply(get_pagerank_value, axis=1)
print('Writing test...')
pagerank_feats_test.to_csv("./data/test_pagerank.csv", index=False)
