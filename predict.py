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
    offline_test = False
    method_set = ['LR', 'RF', 'GBDT', 'ADA', 'XGB']
    method = method_set[4]

    train_path = base + 'train_features.pickle'
    test_path = base + 'test_features.pickle'

    use_vsm = True
    use_magic = True
    use_fuzzy = True
    use_deep = True
    use_deep_bm25 = True
    use_lstm = False
    use_magic_v2 = True

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

    if use_lstm:
        train_lstm = base + 'tune10_train.csv'
        test_lstm = base + 'tune10.csv'

    if use_magic_v2:
        train_magic_v2 = base + 'train_magic_v2.csv'
        test_magic_v2 = base + 'test_magic_v2.csv'

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
        deep_use_index = [0, 1, 2, 3, 6]

    if use_deep_bm25:
        with open(train_deep_bm25, 'rb') as f:
            train_deep_bm25_data = pickle.load(f)
        with open(test_deep_bm25, 'rb') as f:
            test_deep_bm25_data = pickle.load(f)

    if use_lstm:
        train_lstm_data = []
        with open(train_lstm) as f:
            next(f)
            for line in f:
                line = line.split(',')
                train_lstm_data.append(float(line[0]))
        test_lstm_data = []
        with open(test_lstm) as f:
            next(f)
            for line in f:
                line = line.split(',')
                test_lstm_data.append(float(line[0]))

    if use_magic_v2:
        train_magic_v2_data = []
        with open(train_magic_v2) as f:
            next(f)
            for line in f:
                line = line.split(',')
                train_magic_v2_data.append(int(line[1]))
        test_magic_v2_data = []
        with open(test_magic_v2) as f:
            next(f)
            for line in f:
                line = line.split(',')
                test_magic_v2_data.append(int(line[1]))

    print("Initialize features...")

    feature_dict = {'noun_sub': 0, 'verb_sub': 1, 'keyword_match': 2, 'word_difference': 3, 'noun_share': 4, 'verb_share': 5,'keyword_match_ratio': 6}
    feature_dict.update({'bigram_match': 7, 'bigram_match_ratio': 8, 'bigram_difference': 9})
    feature_dict.update({'trigram_match': 10, 'trigram_match_ratio': 11, 'trigram_difference': 12})
    feature_dict.update({'is_same_type': 13})

    allow_features = ['noun_sub', 'verb_sub', 'keyword_match', 'word_difference', 'noun_share', 'verb_share', 'keyword_match_ratio']
    allow_features.extend(['bigram_match', 'bigram_match_ratio', 'bigram_difference'])
    allow_features.extend(['trigram_match', 'trigram_match_ratio', 'trigram_difference'])

    allow_index = [y for x,y in feature_dict.items() if x in allow_features]

    train_features = []
    labels = []

    for i, instance in enumerate(train_data):
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
        if use_lstm:
            instance['features'].append(train_lstm_data[i])
        if use_magic_v2:
            instance['features'].append(train_magic_v2_data[i])
        train_features.append(instance['features'])
        labels.append(instance['is_duplicate'])

    train_features = np.array(train_features)
    labels = np.array(labels)

    test_features = []
    for i, instance in enumerate(test_data):
        instance['features'] = [y for x,y in enumerate(instance['features']) if x in allow_index]
        test_features.append(instance['features'])
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
        if use_lstm:
            instance['features'].append(test_lstm_data[i])
        if use_magic_v2:
            instance['features'].append(test_magic_v2_data[i])
    test_features = np.array(test_features)

    print("Initialize {} model...".format(method))
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
        model = xgboost.XGBClassifier(n_estimators=2000, nthread=18, max_depth=4)
        submit_path = 'submit_xgb.csv'

    # k-fold validation
    if offline_test:
        print("Evaluate k-fold validation...")
        k = 5
        train_features_folds = np.array_split(train_features, k)
        labels_folds = np.array_split(labels, k)
        scores = []
        for i in range(k):
            # features
            X_train = list(train_features_folds)
            X_test = X_train.pop(i)
            X_train = np.concatenate(X_train)
            # labels
            Y_train = list(labels_folds)
            Y_test = Y_train.pop(i)
            Y_train = np.concatenate(Y_train)

            k_score = model.fit(X_train,Y_train).score(X_test,Y_test)
            scores.append(k_score)

        print(mean(scores))
        sys.exit()

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
            value = normalize(prob_result[i])
            f.write(str(i) + ',' + str(value) + '\n')

    print("Finished {} task.".format(method))

