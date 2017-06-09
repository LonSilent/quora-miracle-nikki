from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost
import numpy as np
import pickle
from statistics import mean
import sys
import pandas
import csv

csv.field_size_limit(sys.maxsize)

def normalize(x):
    train_p = 0.37
    real_p = 0.15

    a = real_p / train_p
    b = (1 - real_p) / (1 - train_p)

    return (a * x) / (a * x + b * (1 - x))

def is_nan_or_inf(x):
    return (not np.isfinite(x)) or np.isnan(x)

if __name__ == '__main__':
    base = './data/'
    early_stop = False
    method_set = ['LR', 'RF', 'GBDT', 'ADA', 'XGB', 'GBM']
    method = method_set[5]
    train_skip_row = [3306, 13016, 17682, 20072, 20794, 23305, 23884, 25228, 25315, 39769, 44619, 46596, 47056, 51909, 57484, 63712, 72844, 74304, 86457, 96725, 102512, 104101, 105780, 106577, 108978, 109009, 109311, 115347, 130637, 134403, 139219, 141281, 144506, 151922, 158778, 164553, 169290, 175282, 180461, 181695, 182943, 189396, 189659, 190570, 193246, 193815, 198913, 199110, 201841, 205947, 208199, 208485, 208798, 213220, 216861, 226925, 231151, 231313, 231879, 236655, 245880, 248125, 250701, 254161, 257077, 260779, 263134, 270146, 273065, 289307, 297461, 301583, 306878, 312498, 322705, 324777, 325200, 325530, 326297, 327206, 328601, 328745, 351788, 357127, 361480, 363362, 365317, 381124, 384293, 402423]

    train_path = base + 'train_features.pickle'
    test_path = base + 'test_features.pickle'

    use_vsm = True
    use_magic = True
    use_fuzzy = True
    use_deep = True
    use_deep_bm25 = True
    use_magic_v2 = True
    use_best = True
    use_pagerank = True
    use_glove = True
    use_fasttext = True
    use_bm25_other = True

    if use_vsm:
        train_vsm = base + 'train_vsm.pickle'
        test_vsm = base + 'test_vsm.pickle'

    if use_magic:
        train_magic = base + 'train_magic.csv'
        test_magic = base + 'test_magic.csv'

    if use_fuzzy:
        train_fuzzy = base + 'train_fuzzy.pickle'
        test_fuzzy = base + 'test_fuzzy.pickle'

    if use_deep:
        train_deep = base + 'train_deep.pickle'
        test_deep = base + 'test_deep.pickle'

    if use_deep_bm25:
        train_deep_bm25 = base + 'train_kenter.pickle'
        test_deep_bm25 = base + 'test_kenter.pickle'

    if use_magic_v2:
        train_magic_v2 = './data/new_magic_train.csv'
        test_magic_v2 = './data/new_magic_test.csv'

    if use_best:
        train_best = './data/train_features_01584.csv'
        test_best = './data/test_features_01584.csv'

    if use_pagerank:
        train_pagerank = './data/train_pagerank.csv'
        test_pagerank = './data/test_pagerank.csv'

    if use_glove:
        train_glove = './data/train_glove.pickle'
        test_glove = './data/test_glove.pickle'

    if use_fasttext:
        train_fasttext = './data/train_fasttext.pickle'
        test_fasttext = './data/test_fasttext.pickle'

    if use_bm25_other:
        train_bm25_other = './data/train_bm25_other.pickle'
        test_bm25_other = './data/test_bm25_other.pickle'

    print("Loading files...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    if use_vsm:
        with open(train_vsm, 'rb') as f:
            train_vsm_sim = pickle.load(f)
        with open(test_vsm, 'rb') as f:
            test_vsm_sim = pickle.load(f)

    if use_magic:
        train_magic_data = []
        with open(train_magic) as f:
            next(f)
            for line in f:
                line = [int(x) for x in line.strip().split(',')[1:]]
                train_magic_data.append(line)
        test_magic_data = []
        with open(test_magic) as f:
            next(f)
            for line in f:
                line = [int(x) for x in line.strip().split(',')[1:]]
                test_magic_data.append(line)

    if use_fuzzy:
        with open(train_fuzzy, 'rb') as f:
            train_fuzzy_data = pickle.load(f)
        with open(test_fuzzy, 'rb') as f:
            test_fuzzy_data = pickle.load(f)

    if use_deep:
        with open(train_deep, 'rb') as f:
            train_deep_data = pickle.load(f)
        with open(test_deep, 'rb') as f:
            test_deep_data = pickle.load(f)
        # deep_use_index = [0, 1, 2, 3, 6]
        deep_use_index = [0, 1, 2, 3]

    if use_deep_bm25:
        with open(train_deep_bm25, 'rb') as f:
            train_deep_bm25_data = pickle.load(f)
        with open(test_deep_bm25, 'rb') as f:
            test_deep_bm25_data = pickle.load(f)

    if use_magic_v2:
        train_magic_v2_data = []
        with open(train_magic_v2) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                train_magic_v2_data.append([int(line[1]), float(line[2])])
        test_magic_v2_data = []
        with open(test_magic_v2) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                test_magic_v2_data.append([int(line[1]), float(line[2])])

    if use_best:
        best_cols = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        train_best_data = []
        with open(train_best) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                for i in range(len(line)):
                    if line[i] == '':
                        line[i] = np.nan
                line = [float(y) for x,y in enumerate(line) if x in best_cols]
                train_best_data.append(line)
        test_best_data = []
        with open(test_best) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                for i in range(len(line)):
                    if line[i] == '':
                        line[i] = np.nan
                line = [float(y) for x,y in enumerate(line) if x in best_cols]
                test_best_data.append(line)

    if use_pagerank:
        train_pagerank_data = []
        with open(train_pagerank) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                train_pagerank_data.append([float(line[0]), float(line[1])])
        test_pagerank_data = []
        with open(test_pagerank) as f:
            next(f)
            for line in f:
                line = line.strip().split(',')
                test_pagerank_data.append([float(line[0]), float(line[1])])

    if use_glove:
        with open(train_glove, 'rb') as f:
            train_glove_data = pickle.load(f)
        with open(test_glove, 'rb') as f:
            test_glove_data = pickle.load(f)

    if use_fasttext:
        with open(train_fasttext, 'rb') as f:
            train_fasttext_data = pickle.load(f)
        with open(test_fasttext, 'rb') as f:
            test_fasttext_data = pickle.load(f)

    if use_bm25_other:
        with open(train_bm25_other, 'rb') as f:
            train_bm25_other_data = pickle.load(f)
        with open(test_bm25_other, 'rb') as f:
            test_bm25_other_data = pickle.load(f)        

    print("Initialize features...")

    feature_dict = {'noun_sub': 0, 'verb_sub': 1, 'keyword_match': 2, 'word_difference': 3, 'noun_share': 4, 'verb_share': 5,'keyword_match_ratio': 6}
    feature_dict.update({'bigram_match': 7, 'bigram_match_ratio': 8, 'bigram_difference': 9})
    feature_dict.update({'trigram_match': 10, 'trigram_match_ratio': 11, 'trigram_difference': 12})

    allow_features = ['noun_sub', 'verb_sub', 'keyword_match', 'word_difference', 'noun_share', 'verb_share', 'keyword_match_ratio']
    allow_features.extend(['bigram_match', 'bigram_match_ratio', 'bigram_difference'])
    allow_features.extend(['trigram_match', 'trigram_match_ratio', 'trigram_difference'])

    allow_index = [y for x,y in feature_dict.items() if x in allow_features]

    train_features = []
    labels = []
    for i, instance in enumerate(train_data):
        if i in train_skip_row:
            continue
        instance['features'] = [y for x,y in enumerate(instance['features']) if x in allow_index]
        if use_vsm:
            instance['features'].append(train_vsm_sim[i])
        if use_magic:
            instance['features'].extend(train_magic_data[i])
        if use_fuzzy:
            instance['features'].extend(train_fuzzy_data[i][:-1])
        if use_deep:
            for j in range(len(train_deep_data[i])):
                if is_nan_or_inf(train_deep_data[i][j]):
                    train_deep_data[i][j] = 1000
            train_deep_data[i] = [y for x,y in enumerate(train_deep_data[i]) if x in deep_use_index]
            instance['features'].extend(train_deep_data[i])
        if use_deep_bm25:
            instance['features'].append(train_deep_bm25_data[i])
        if use_magic_v2:
            instance['features'].extend(train_magic_v2_data[i])
        if use_best:
            instance['features'].extend(train_best_data[i])
        if use_pagerank:
            instance['features'].extend(train_pagerank_data[i])
        if use_glove:
            for j in range(len(train_glove_data[i])):
                if is_nan_or_inf(train_glove_data[i][j]):
                    train_glove_data[i][j] = 1000
            train_glove_data[i] = [y for x,y in enumerate(train_glove_data[i]) if x in deep_use_index]
            instance['features'].extend(train_glove_data[i])
        if use_fasttext:
            for j in range(len(train_fasttext_data[i])):
                if is_nan_or_inf(train_fasttext_data[i][j]):
                    train_fasttext_data[i][j] = 1000
            train_fasttext_data[i] = [y for x,y in enumerate(train_fasttext_data[i]) if x in deep_use_index]
            instance['features'].extend(train_fasttext_data[i])
        if use_bm25_other:
            instance['features'].extend(train_bm25_other_data[i])
        train_features.append(instance['features'])
        labels.append(instance['is_duplicate'])

    train_features = np.array(train_features)
    labels = np.array(labels)
    # sys.exit()

    test_features = []
    for i, instance in enumerate(test_data):
        instance['features'] = [y for x,y in enumerate(instance['features']) if x in allow_index]
        if use_vsm:
            instance['features'].append(test_vsm_sim[i])
        if use_magic:
            instance['features'].extend(test_magic_data[i])
        if use_fuzzy:
            instance['features'].extend(test_fuzzy_data[i][:-1])
        if use_deep:
            for j in range(len(test_deep_data[i])):
                if is_nan_or_inf(test_deep_data[i][j]):
                    test_deep_data[i][j] = 1000
            test_deep_data[i] = [y for x,y in enumerate(test_deep_data[i]) if x in deep_use_index]
            instance['features'].extend(test_deep_data[i])
        if use_deep_bm25:
            instance['features'].append(test_deep_bm25_data[i])
        if use_magic_v2:
            # instance['features'].append(test_magic_v2_data[i])
            instance['features'].extend(test_magic_v2_data[i])
        if use_best:
            instance['features'].extend(test_best_data[i])
        if use_pagerank:
            instance['features'].extend(test_pagerank_data[i])
        if use_glove:
            for j in range(len(test_glove_data[i])):
                if is_nan_or_inf(test_glove_data[i][j]):
                    test_glove_data[i][j] = 1000
            test_glove_data[i] = [y for x,y in enumerate(test_glove_data[i]) if x in deep_use_index]
            instance['features'].extend(test_glove_data[i])
        if use_fasttext:
            for j in range(len(test_fasttext_data[i])):
                if is_nan_or_inf(test_fasttext_data[i][j]):
                    test_fasttext_data[i][j] = 1000
            test_fasttext_data[i] = [y for x,y in enumerate(test_fasttext_data[i]) if x in deep_use_index]
            instance['features'].extend(test_fasttext_data[i])
        if use_bm25_other:
            instance['features'].extend(test_bm25_other_data[i])
        test_features.append(instance['features'])

    test_features = np.array(test_features)

    print("Training {} model...".format(method))
    if method == 'LR':
        model = LogisticRegression(n_jobs=-1)
        submit_path = 'submit_logistic.csv'
    elif method == 'RF':
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1)
        submit_path = 'submit_rf.csv'
    elif method == 'GBDT':
        model = GradientBoostingClassifier(n_estimators=1200)
        submit_path = 'submit_gbdt.csv'
    elif method == 'ADA':
        model = AdaBoostClassifier(n_estimators=100)
        submit_path = 'submit_ada.csv'
    elif method == 'XGB':
        model = xgboost.XGBClassifier(n_estimators=2000, nthread=20, max_depth=4, subsample=0.8)
        submit_path = 'submit_xgb.csv'
    elif method == 'GBM':
        model = GBMClassifier(exec_path='/home/bschang/LightGBM/lightgbm', num_leaves=2000)
        submit_path = 'submit_gbm.csv'

    if early_stop:
        k = 10
        train_features_folds = np.array_split(train_features, k)
        labels_folds = np.array_split(labels, k)
        # features
        X_train = list(train_features_folds)
        X_test = X_train.pop(k-1)
        X_train = np.concatenate(X_train)
        # labels
        Y_train = list(labels_folds)
        Y_test = Y_train.pop(k-1)
        Y_train = np.concatenate(Y_train)
        eval_set = [(X_test, Y_test)]

        model.fit(X_train, Y_train, eval_metric='logloss', eval_set=eval_set, early_stopping_rounds=100)
    else:
        model.fit(train_features, labels)

    print('Feature importances:')
    print(model.feature_importances_)
    print("predicting...")
    prob_result = []
    prob_result = model.predict_proba(test_features)

    print("Finished predicting.")

    prob_result = [x[1] for x in prob_result]

    print("Writing result...")
    with open(submit_path, 'w') as f:
        f.write('test_id,is_duplicate\n')
        for i in range(len(test_data)):
            # value = prob_result[i]
            value = normalize(prob_result[i])
            f.write(str(i) + ',' + str(value) + '\n')

    print("Finished {} task.".format(method))

