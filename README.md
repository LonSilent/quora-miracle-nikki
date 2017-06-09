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

## Feature Description

```python
allow_features = ['noun_sub', 'verb_sub', 'keyword_match', 'word_difference', 
'noun_share', 'verb_share', 'keyword_match_ratio', 'bigram_match', 
'bigram_match_ratio', 'bigram_difference', 'trigram_match', 'trigram_match_ratio', 'trigram_difference', 
'bm25_cosine', 'q1_freq',  'q2_freq',
'qratio', 'wratio', 'ratio', 'partial_ratio', 'partial_token_sort_ratio', 
'token_set_ratio', 'token_sort_ratio',
'wmd', 'norm_wmd', 'sent2vec_cosine', 'sent2vec_cityblock', 'deep_bm25',
'q1_q2_intersect', 'q1_q2_wm_ratio',
'word_match' , 'tfidf_wm' , 'tfidf_wm_stops' ,'jaccard' , 
'wc_diff', 'wc_ratio', 'wc_diff_unique', 'wc_ratio_unique', 
'wc_diff_unq_stop', 'wc_ratio_unique_stop', 'same_start_word', 
'char_diff', 'char_diff_unq_stop', 'total_unique_words', 
'total_unq_words_stop', 'char_ratio', 'q_type1', 'q_type2',
'q1_pagerank', 'q2_pagerank', 
'glove_wmd', 'glove_norm_wmd', 'glove_sent2vec_cosine', 'glove_sent2vec_cityblock',
'fasttext_wmd', 'fasttext_norm_wmd', 'fasttext_sent2vec_cosine', 'fasttext_sent2vec_cityblock',
'glove_deep_bm25', 'fasttext_deep_bm25']
```


## Get Deeper
Please go to official [word2vec website](https://code.google.com/archive/p/word2vec/) on Google, and download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). Next, you should modify `model_path` variable in `deep_feature.py`.
