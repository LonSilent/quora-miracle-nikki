# quora-miracle-nikki
kaggle competition for wsm final project

## prepare your data
put `train.csv` and `test.csv` in `./data` directory

## dependecies
`numpy`, `pandas`, `nltk`, `gensim`, `fuzzywuzzy`, `scikit-learn`, `xgboost`, only test on python3

## file description
```
preprocess.py: basic statistic features
const.py: some dict to do preprocessing
tfidf.py: bm25 cosine
fuzzy.py: fuzzy string similarity
magic.py: questions' duplicated features
d2v.py: Doc2Vec similarity (not improved)
```
```
predict.py: concat features, and train model to predict 
(now GBDT in sklearn or xgboost will get best result)
```
