# code from https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain/comments

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

def q1_q2_intersect(row):
    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

train_orig =  pd.read_csv('./data/train.csv', header=0)
test_orig =  pd.read_csv('./data/test.csv', header=0)

ques = pd.concat([train_orig[['question1', 'question2']], test_orig[['question1', 'question2']]], axis=0).reset_index(drop='index')


q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

train_orig['q1_q2_intersect'] = train_orig.apply(q1_q2_intersect, axis=1, raw=True)
test_orig['q1_q2_intersect'] = test_orig.apply(q1_q2_intersect, axis=1, raw=True)

train_feat = train_orig[['q1_q2_intersect']]
test_feat = test_orig[['q1_q2_intersect']]

train_feat.to_csv('./data/train_magic_v2.csv')
test_feat.to_csv('./data/test_magic_v2.csv')
