import csv
import timeit
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from const import *
import sys

NOUN = ['NN','NNP','NNS','JJ']
VERB = ['VB','VBD','VBG','VBN','VBP','VBZ']
symbol = ['...',':','?', '(' ,')',',','[',']','{','}']

def num_of_noun(tokens):
    return len([x for (x,y) in tokens if y in NOUN])

def num_of_verb(tokens):
    return len([x for (x,y) in tokens if y in VERB])

def noun_list(tokens):
    return [x for (x,y) in tokens if y in NOUN]

def verb_list(tokens):
    return [x for (x,y) in tokens if y in VERB]

if __name__ == '__main__':
    base = './data/'
    is_train = True
    if is_train:
        file_path = base + 'train.csv'
        output_path = base + 'train.pickle'
        result_path = base + 'train_features.pickle'
    else:
        file_path = base + 'test.csv'
        output_path = base + 'test.pickle'
        result_path = base + 'test_features.pickle'

    train = []
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    start_time = timeit.default_timer()

    with open(file_path) as f:
        columns = f.readline().strip().replace('\"','').split(',')
        reader = csv.reader(f)
        for row in reader:
            train_item = {}
            for i, item in enumerate(row):
                for key, value in ABBR_DICT.items():
                    train_item[columns[i]] = row[i].lower().replace(key, value)
            train.append(train_item)

    for item in train:
        item['question1'] = [lemmatizer.lemmatize(x.lower()) for x in nltk.word_tokenize(item['question1']) if x not in symbol]
        item['question2'] = [lemmatizer.lemmatize(x.lower()) for x in nltk.word_tokenize(item['question2']) if x not in symbol]
    with open(output_path, 'wb') as f:
        pickle.dump(train, f)

    # sys.exit()

    print("Loading pickle files...")
    with open(output_path, 'rb') as f:
        train = pickle.load(f)
    print("Finished Loading.")
    result = []
    
    print("Extracting features...")
    for index, instance in enumerate(train):
        q1 = instance['question1']
        q2 = instance['question2']
        # print(q1)
        # print(q2)
        item = {}
        features = []

        keyword_match = len(set.intersection(set(q1), set(q2)))
        keyword_match_ratio = keyword_match / max( len(set.union(set(q1), set(q2))), 1)
        word_difference = len(set.symmetric_difference(set(q1), set(q2)))

        bigram_q1 = [x for x in ngrams(q1, 2)]
        bigram_q2 = [x for x in ngrams(q2, 2)]
        bigram_match = len(set.intersection(set(bigram_q1), set(bigram_q2)))
        bigram_match_ratio = bigram_match / max( len(set.union(set(bigram_q1), set(bigram_q2))), 1)
        bigram_difference = len(set.symmetric_difference(set(bigram_q1), set(bigram_q2)))

        trigram_q1 = [x for x in ngrams(q1, 3)]
        trigram_q2 = [x for x in ngrams(q2, 3)]
        trigram_match = len(set.intersection(set(trigram_q1), set(trigram_q2)))
        trigram_match_ratio = trigram_match / max( len(set.union(set(trigram_q1), set(trigram_q2))), 1)
        trigram_difference = len(set.symmetric_difference(set(trigram_q1), set(trigram_q2)))

        pos_q1 = nltk.pos_tag(q1)
        pos_q2 = nltk.pos_tag(q2)

        noun_sub = abs(num_of_noun(pos_q1) - num_of_noun(pos_q2))
        verb_sub = abs(num_of_verb(pos_q1) - num_of_verb(pos_q2))

        noun_share = len(set.intersection(set( noun_list(pos_q1) ), set(noun_list(pos_q2) )))
        verb_share = len(set.intersection(set( verb_list(pos_q1) ), set(verb_list(pos_q2) )))

        if len(q1) == 0 or len(q1) == 0:
            is_same_type = 0
        elif q1[0] != q1[0]:
            is_same_type = 0
        else:
            is_same_type = 1

        features.extend([noun_sub, verb_sub, keyword_match, word_difference, noun_share, verb_share, keyword_match_ratio])
        features.extend([bigram_match, bigram_match_ratio, bigram_difference])
        features.extend([trigram_match, trigram_match_ratio, trigram_difference])
        features.extend([is_same_type])

        item['features'] = features
        if is_train:
            item['is_duplicate'] = instance['is_duplicate']
        result.append(item)
        print('{}/{}'.format(index, len(train)), end='\r')

    print("Start dump.")
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)


    stop_time = timeit.default_timer()
    print("Finished, and cost {0:.2f} secs".format(stop_time-start_time))
