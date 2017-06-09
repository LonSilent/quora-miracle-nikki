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

if __name__ == '__main__':
    early_stop = False
    train_feature_path = './train_features_best.csv'
    test_feature_path = './test_features_best.csv'
    label_path = './labels.txt'

    method_set = ['XGB', 'GBM']
    method = method_set[0]

    print("Initialize train features...")

    train_features = []
    with open(train_feature_path) as f:
        for line in f:
            line = line.strip().split(',')
            for i in range(len(line)):
                if line[i] == '':
                    line[i] = np.nan
            line = [float(x) for x in line]
            train_features.append(line)

    print("Initialize test features...")

    test_features = []
    with open(test_feature_path) as f:
        for line in f:
            line = line.strip().split(',')
            for i in range(len(line)):
                if line[i] == '':
                    line[i] = np.nan
            line = [float(x) for x in line]
            test_features.append(line)

    print("Initialize labels...")

    labels = []
    with open(label_path) as f:
        for line in f:
            labels.append(int(line.strip()))

    len_test_data = len(test_features)
    train_features = np.array(train_features)
    labels = np.array(labels)
    test_features = np.array(test_features)

    print("Training {} model...".format(method))
    if method == 'XGB':
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

    # print('Feature importances:')
    # print(model.feature_importances_)
    print("predicting...")
    prob_result = []
    prob_result = model.predict_proba(test_features)

    print("Finished predicting.")

    prob_result = [x[1] for x in prob_result]

    print("Writing result...")
    with open(submit_path, 'w') as f:
        f.write('test_id,is_duplicate\n')
        for i in range(len_test_data):
            # value = prob_result[i]
            value = normalize(prob_result[i])
            f.write(str(i) + ',' + str(value) + '\n')

    print("Finished {} task.".format(method))


