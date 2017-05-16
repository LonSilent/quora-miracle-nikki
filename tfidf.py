from collections import defaultdict, Counter
import pickle
import itertools
from math import log10, sqrt

def get_tf(tokens):
    return dict(Counter(tokens))

def get_df(corpus):
    df = defaultdict(int)
    for tokens in corpus:
        for word, count in tokens.items():
            df[word] += 1
    return df

def get_idf(df, corpus_count):
    idf = {}
    for word, doc_freq in df.items():
        idf[word] = 1 + log10(corpus_count / doc_freq)

    return idf

def get_vector(tokens, idf, avg_len):
    vector = {}
    doc_len = sum(list(tokens.values()))
    for word, count in tokens.items():
        vector[word] = bm25(idf[word], count, doc_len, avg_len)
        # vector[word] = count * idf[word]
    return vector


def bm25(idf, tf, doc_len, avg_len):
    b = 0.75
    k = 1.2

    return idf * (tf * (k + 1)) / (tf + k * (1 - b + b * (doc_len / avg_len))) 

def cosine(v1, v2):
    dot = 0
    for word, score in v1.items():
        if word in v2:
            dot += score * v2[word]
    if dot == 0:
        return 0
    else:
        return dot / ( norm(v1.values()) * norm(v2.values()) )

def norm(v):
    return sqrt(sum(i**2 for i in v)) 

if __name__ == '__main__':
    train_file = 'train.pickle'
    test_file = 'test.pickle'

    # outfile = 'train_vsm.pickle'
    outfile = 'test_vsm.pickle'

    print("Loading files...")
    with open(train_file, 'rb') as f:
        train = pickle.load(f)
    with open(test_file, 'rb') as f:
        test = pickle.load(f)

    train_corpus = []
    test_corpus = []

    print('Calculate TF...')
    for instance in train:
        tf1 = get_tf(instance['question1'])
        tf2 = get_tf(instance['question2'])
        train_corpus.append([tf1, tf2])

    for instance in test:
        tf1 = get_tf(instance['question1'])
        tf2 = get_tf(instance['question2'])
        test_corpus.append([tf1, tf2])

    print('Calculate IDF...')
    all_corpus = list(itertools.chain.from_iterable(train_corpus)) + list(itertools.chain.from_iterable(test_corpus))
    # print(all_corpus[:10])
    corpus_count = len(all_corpus)
    df = get_df(all_corpus)
    idf = get_idf(df, corpus_count)

    avg_len = 0
    for tokens in all_corpus:
        count_tokens = 0
        for word, count in tokens.items():
            count_tokens += count
        avg_len += count_tokens
    avg_len /= corpus_count

    result = []
    print('Calculate Cosine...')
    if 'train' in outfile:
        for pair in train_corpus:
            vec1 = get_vector(pair[0], idf, avg_len)
            vec2 = get_vector(pair[1], idf, avg_len)
            cos_score = cosine(vec1, vec2)
            result.append(cos_score)
    else:
        for pair in test_corpus:
            vec1 = get_vector(pair[0], idf, avg_len)
            vec2 = get_vector(pair[1], idf, avg_len)
            cos_score = cosine(vec1, vec2)
            result.append(cos_score)

    with open(outfile, 'wb') as f:
        pickle.dump(result, f)




