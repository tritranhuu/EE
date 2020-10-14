import numpy as np
import tensorflow as tf
import itertools
import re

from bratreader.repomodel import RepoModel

class DataProcessor():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.corpus = RepoModel(data_dir)
        self.documents = self.corpus.documents
        self.sentences = self.load_sentences()

    def load_sentences(self):
        self.sentences = []
        for doc in self.documents:
            sentences = self.documents[doc].sentences      
            for sent in sentences:
                entities = []
                words = []
                labels = []
                for w in sent.words:
                    words.append(w.form)
                    anno = w.annotations
                    entity = 'O'
                    if len(anno) > 0:
                        if not (w.annotations[0].label.isupper()):
                            labels.append(anno[0].label)
                        else:
                            
                            labels.append('O')
                            entity = w.annotations[0].label
                    else:
                        labels.append('O')
                    entities.append(entity)
                label_set = list(set(labels))
                if len(label_set) > 1:
                    self.sentences.append({'words': words, 'entities': entities, 'labels': labels})
        return self.sentences


    def get_train_examples(self):
        pass

    def get_dev_examples(self):
        pass

    def get_test_examples(self):
        pass

    def get_sentences(self):
        pass