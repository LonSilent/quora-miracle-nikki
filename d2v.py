import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import utils

class DocumentsWithIDs(object):
    def __init__(self, source, ids):
        self.source = source
        self.ids = ids

    def __iter__(self):
        for i, article in enumerate(self.source):
            yield TaggedDocument(utils.to_unicode(article).strip().split(), [self.ids[i]])

def train_doc2vec(corpus, model_path):
    ids = [str(x) for x in range(len(corpus))]

    sentences = DocumentsWithIDs(corpus, ids)
    model = Doc2Vec(min_count=0, workers=18, alpha=0.025, min_alpha=0.025, size=500)
    model.build_vocab(sentences)
    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        print('Finish epoch {}'.format(epoch))

    model.save(model_path)

if __name__ == '__main__':
    is_train = True
    not_has_model = False
    train_path = 'train.pickle'
    test_path = 'test.pickle'

    model_path = '/tmp2/bschang/d2v.model'
    output_path = 'train_d2v.pickle'
    # output_path = 'test_d2v.pickle'

    with open(train_path, 'rb') as f:
        train = pickle.load(f)
    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    train_offset = 0
    all_corpus = []
    result = []

    for instance in train:
        all_corpus.append(' '.join(instance['question1']))
        all_corpus.append(' '.join(instance['question2']))
        train_offset += 2

    test_offset = train_offset
    for instance in test:
        all_corpus.append(' '.join(instance['question1']))
        all_corpus.append(' '.join(instance['question2']))
        test_offset += 2

    if not_has_model:
        train_doc2vec(all_corpus, model_path)

    model = Doc2Vec.load(model_path)
    if 'train' in output_path:
        for i in range(0, train_offset, 2):
            doc1 = str(i)
            doc2 = str(i + 1)
            sim = model.docvecs.similarity(doc1, doc2)
            result.append(sim)
        print(result[:10])
        with open('train_d2v.pickle', 'wb') as f:
            pickle.dump(result, f)

    else:
        for i in range(train_offset, test_offset, 2):
            doc1 = str(i)
            doc2 = str(i + 1)
            sim = model.docvecs.similarity(doc1, doc2)
            result.append(sim)
        print(result[:10])
        with open('test_d2v.pickle', 'wb') as f:
            pickle.dump(result, f)


    

