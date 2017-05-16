import pickle
import fuzzywuzzy.fuzz as fuzzy

if __name__ == '__main__':
    data_path = './data/train.pickle'
    output_path = './data/train_fuzzy.pickle'

    data_path = './data/test.pickle'
    output_path = './data/test_fuzzy.pickle'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    result = []
    for i in range(len(data)):
        # print(train[5])
        sentence_q1 = ' '.join(data[i]['question1'])
        sentence_q2 = ' '.join(data[i]['question2'])
        # print(sentence_q1)
        # print(sentence_q2)

        qratio = fuzzy.QRatio(sentence_q1, sentence_q2)
        wratio = fuzzy.WRatio(sentence_q1, sentence_q2)
        ratio = fuzzy.ratio(sentence_q1, sentence_q2)
        partial_ratio = fuzzy.partial_ratio(sentence_q1, sentence_q2)
        partial_token_set_ratio = fuzzy.partial_token_set_ratio(sentence_q1, sentence_q2)
        partial_token_sort_ratio = fuzzy.partial_token_sort_ratio(sentence_q1, sentence_q2)
        token_set_ratio = fuzzy.token_set_ratio(sentence_q1, sentence_q2)
        token_sort_ratio = fuzzy.token_sort_ratio(sentence_q1, sentence_q2)

        fuzzyee = [qratio, wratio, ratio, partial_ratio, partial_token_sort_ratio, token_set_ratio, token_sort_ratio]

        result.append(fuzzyee)
        print('{}/{}'.format(i, len(data)), end='\r')


    with open(output_path, 'wb') as f:
        pickle.dump(result, f)


