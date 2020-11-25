import gensim
from collections import Counter
import torch
from torchtext.data import Field, BucketIterator, NestedField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab


class Corpus(object):

  def __init__(self, args):
    # list all the fields
    self.word_field = Field(lower=True)
    self.event_field = Field(unk_token=None)
    self.entity_field = Field(unk_token=None)
    self.argument_field = Field(unk_token=None)
    self.trigger_pos_field = Field(unk_token=None)
    self.char_nesting_field = Field(tokenize=list)
    self.char_field = NestedField(self.char_nesting_field)

    self.wv = args.wv_file
    # create dataset using built-in parser from torchtext
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=args.input_folder,
        train="train.txt",
        validation="dev.txt",
        test="test.txt",
        fields=(
          (("word", "char"), (self.word_field, self.char_field)), 
          ("event", self.event_field),
          ("entity", self.entity_field),
          ("argument", self.argument_field),
          ("trigger_pos", self.trigger_pos_field)),
    )
    # convert fields to vocabulary list
    # self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.event_field.build_vocab(self.train_dataset.event)
    # create iterator for batch input
    
    
    if args.wv_file:
        print("start loading embedding")
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format(args.wv_file, binary=False)
        print("done loading embedding")
        self.embedding_dim = self.wv_model.vector_size
        word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
        word_counter = Counter(word_freq)
        self.word_field.vocab = Vocab(word_counter, min_freq=args.min_word_freq)
            # mapping each vector/embedding from word2vec model to word_field vocabs
        vectors = []
        print("start loading vec", len(self.word_field.vocab.stoi))
        for word, idx in self.word_field.vocab.stoi.items():
            if word in self.wv_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.embedding_dim))
        print("done loading vec")
        del self.wv_model
        self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                # list of vector embedding, orderred according to word_field.vocab
                vectors=vectors,
                dim=self.embedding_dim
            )
        
    else:
        self.word_field.build_vocab(self.train_dataset.word, min_freq=args.min_word_freq)
    self.char_field.build_vocab(self.train_dataset.char)
    self.entity_field.build_vocab(self.train_dataset.entity)
    self.argument_field.build_vocab(self.train_dataset.argument)
    self.trigger_pos_field.build_vocab(self.train_dataset.trigger_pos)

    self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
        batch_size=args.batch_size,
        shuffle=False,
    ) 
    
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.event_pad_idx = self.event_field.vocab.stoi[self.event_field.pad_token]
    self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
    self.entity_pad_idx = self.entity_field.vocab.stoi[self.entity_field.pad_token]
    self.argument_pad_idx = self.entity_field.vocab.stoi[self.entity_field.pad_token]
    
