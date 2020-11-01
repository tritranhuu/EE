import gensim
from collections import Counter
import torch
from torchtext.data import Field, BucketIterator, NestedField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab

class Corpus(object):

  def __init__(self, 
    # input_folder, min_word_freq, batch_size, wv_file=None, 
               args
    ):
    # list all the fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)
    self.char_nesting_field = Field(tokenize=list)
    self.char_field = NestedField(self.char_nesting_field)

    # create dataset using built-in parser from torchtext
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=args.input_folder,
        train="train.csv",
        validation="dev.csv",
        test="test.csv",
        fields=(
          (("word", "char"), (self.word_field, self.char_field)), 
          ("tag", self.tag_field))
    )
    # convert fields to vocabulary list
    # self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)
    # create iterator for batch input
    
    
    if args.wv_file:
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format(args.wv_file, binary=True)
        self.embedding_dim = self.wv_model.vector_size
        word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
        word_counter = Counter(word_freq)
        self.word_field.vocab = Vocab(word_counter, min_freq=args.min_word_freq)
            # mapping each vector/embedding from word2vec model to word_field vocabs
        vectors = []
        for word, idx in self.word_field.vocab.stoi.items():
            if word in self.wv_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.embedding_dim))
        self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                # list of vector embedding, orderred according to word_field.vocab
                vectors=vectors,
                dim=self.embedding_dim
            )
    else:
        self.word_field.build_vocab(self.train_dataset.word, min_freq=args.min_word_freq)
    self.char_field.build_vocab(self.train_dataset.char)

    self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
        batch_size=args.batch_size
    ) 
    
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
    self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]