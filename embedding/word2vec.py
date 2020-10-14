from gensim.models.word2vec import Word2Vec
import gensim
from embedding.random_vec import RandomVec
import pickle as pkl

class WordVec:
    def __init__(self, args):
        if args.restore is None:
            corpus = open(args.corpus, 'r').read().lower().split()
            sentences = []
            sentence = []
            length = 0
            for word in corpus:
                sentence.append(word)
                length += 1
                if length == args.sentence_length:
                    sentences.append(sentence)
                    sentence = []
                    length = 0
            if length != 0:
                sentences.append(sentence)
            print('training embedding')
            self.wvec_model = Word2Vec(sentences=sentences, size=args.dimension, window=args.window,
                                       workers=args.workers,
                                       sg=args.sg,
                                       batch_words=args.batch_size, min_count=1, max_vocab_size=args.vocab_size)
        else:
            self.wvec_model = gensim.models.KeyedVectors.load_word2vec_format(args.restore, binary=True)
        self.rand_model = RandomVec(args.dimension)
    
    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            return self.rand_model[word]