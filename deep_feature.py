import gensim
from nltk.corpus import stopwords
import pickle
import numpy as np
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

stop = stopwords.words('english')

def wmd(s1, s2, model):
    s1 = [w for w in s1 if w not in stop]
    s2 = [w for w in s2 if w not in stop]
    return model.wmdistance(s1, s2)

def sent2vec(sentence, model):
    sentence = [w for w in sentence if w.isalpha()]
    M = []
    for word in sentence:
        try:
            M.append(model[word])
        except:
            continue
    M = np.array(M)
    vec = M.sum(axis=0)

    return vec / np.sqrt((vec**2).sum())

if __name__ == '__main__':
    # model_type = 'deep'
    # model_type = 'glove'
    model_type = 'fasttext'

    train_path = './data/train.pickle'
    test_path = './data/test.pickle'

    if model_type == 'deep': 
        model_path = '/tmp2/bschang/GoogleNews-vectors-negative300.bin.gz'
    elif model_type == 'glove':
        model_path = '/tmp2/bschang/glove-word2vec.txt'
    elif model_type == 'fasttext':
        model_path = '/tmp2/bschang/fasttext-word2vec.txt'
    
    data_path = train_path
    output_path = './data/train_{}.pickle'.format(model_type)

    # data_path = test_path
    # output_path = './data/test_{}.pickle'.format(model_type)

    print('Loading pickle...')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print('Loading word2vec file...')
    if model_type == 'deep': 
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

    print('Calculate wmd...')
    wmd_list = []
    for i in range(len(data)):
        q1 = data[i]['question1']
        q2 = data[i]['question2']
        wmd_list.append(wmd(q1, q2, model))

    print('Loading normalize word2vec file...')
    if model_type == 'deep': 
        norm_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        norm_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    norm_model.init_sims(replace=True)
    
    print('Calculate normalize wmd...')
    norm_wmd_list = []
    for i in range(len(data)):
        q1 = data[i]['question1']
        q2 = data[i]['question2']
        norm_wmd_list.append(wmd(q1, q2, norm_model))

    result = [[x,y] for x,y in zip(wmd_list, norm_wmd_list)]

    print('Calculate dist features...')
    question1_vecs = np.zeros((len(data), 300))
    question2_vecs = np.zeros((len(data), 300))

    for i in range(len(data)):
        q1 = data[i]['question1']
        q2 = data[i]['question2']

        if len(q1) == 0 or len(q2) == 0:
            cosine_dist = 1000
            cityblock_dist = 1000
            jaccard_dist = 1000
            canberra_dist = 1000
            euclidean_dist = 1000
            minkowski_dist = 1000
            braycurtis_dist = 1000
            dist_features = [cosine_dist, cityblock_dist, jaccard_dist, canberra_dist, euclidean_dist, minkowski_dist, braycurtis_dist]
            result[i].extend(dist_features)
            print('{}/{}'.format(i, len(data)), end='\r')
            continue

        question1_vecs[i, :] = sent2vec(q1, model)
        question2_vecs[i, :] = sent2vec(q2, model)

        cosine_dist = cosine(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        cityblock_dist = cityblock(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        jaccard_dist = jaccard(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        canberra_dist = canberra(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        euclidean_dist = euclidean(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        minkowski_dist = minkowski(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]), 3)
        braycurtis_dist = braycurtis(np.nan_to_num(question1_vecs[i, :]), np.nan_to_num(question2_vecs[i, :]))
        
        dist_features = [cosine_dist, cityblock_dist, jaccard_dist, canberra_dist, euclidean_dist, minkowski_dist, braycurtis_dist]
        result[i].extend(dist_features)
        # print('{}/{}'.format(i, len(data)), end='\r')

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
