# quora-miracle-nikki
kaggle competition for wsm final project

## Prepare Your Data
put `train.csv` and `test.csv` in `./data` directory

## Dependecies
`numpy`, `pandas`, `nltk`, `gensim`, `fuzzywuzzy`, `python-Levenshtein`, `scikit-learn`, `xgboost`, only test on python3

## File Description
Feature engineering:
```
preprocess.py: basic statistic features
const.py: some dict to do preprocessing
tfidf.py: bm25 cosine
fuzzy.py: fuzzy string similarity
magic.py: questions' duplicated features
d2v.py: Doc2Vec similarity (not improved)
deep_feature.py: Word2Vec features
bm25-word2vec.py: implementation of the paper [CIKM’15, Short Text Similarity with Word Embeddings]
magic_v2.py: questions' hash intersection count
```
Classification model:
```
predict.py: concat features, and train model to predict 
(now GBDT in sklearn or xgboost will get best result)
```

## Get Deeper
Please go to official [word2vec website](https://code.google.com/archive/p/word2vec/) on Google, and download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). Next, you should modify `model_path` variable in `deep_feature.py`.
