import os
import time
import gensim
import math
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torchcrf import CRF
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

class Embeddings(nn.Module):

    def __init__(self,
                 word_input_dim,
                 word_emb_dim,
                 word_emb_pretrained,
                 word_emb_dropout,
                 word_emb_froze,
                 use_char_emb,
                 char_input_dim,
                 char_emb_dim,
                 char_emb_dropout,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 char_cnn_dropout,
                 entity_emb_size=None,
                 entity_emb_dim=0,
                 entity_emb_dropout=0,
                 word_pad_idx,
                 char_pad_idx,
                 entity_pad_idx,
                 device
        ):
        super().__init__()
        self.device = device
        self.word_pad_idx = word_pad_idx
        self.char_pad_idx = char_pad_idx
        #Initialize embedding if pretrained is given
        if word_emb_pretrained is not None:
            self.word_emb = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(word_emb_pretrained),
                padding_idx = self.word_pad_idx,
                freeze=word_emb_froze
            )
        else:
            self.word_emb = nn.Embedding(
                num_embeddings=word_input_dim,
                embedding_dim=word_emb_dim,
                padding_idx=self.word_pad_idx
            )
            self.word_emb.weight.data[self.word_pad_idx] = torch.zeros(word_emb_dim)
        self.word_emb_dropout = nn.Dropout(word_emb_dropout)
        self.output_dim = word_emb_dim

        self.use_char_emb = use_char_emb
        if self.use_char_emb:
            self.char_emb_dim = char_emb_dim
            self.char_emb = nn.Embedding(
                num_embeddings=char_input_dim,
                embedding_dim=char_emb_dim,
                padding_idx=char_pad_idx
            )
            self.char_emb.weight.data[self.char_pad_idx]=torch.zeros(self.char_emb_dim)
            self.char_emb_dropout = nn.Dropout(char_emb_dropout)
            #CharCNN
            self.char_cnn = nn.Conv1d(
                in_channels = char_emb_dim,
                out_channels=char_emb_dim*char_cnn_filter_num,
                kernel_size=char_cnn_kernel_size,
                groups=char_emb_dim
            )
            self.char_cnn_dropout = nn.Dropout(char_cnn_dropout)
            self.output_dim += char_emb_dim * char_cnn_filter_num
        self.entity_emb_size = entity_emb_size
        if self.entity_emb_size != None:
            self.entity_emb_dim = entity_emb_dim
            self.entity_emb = nn.Embedding(
                num_embeddings=entity_emb_size,
                out_channels=entity_emb_dim
            )
            self.output_dim += entity_emb_dim
            self.entity_emb_dropout = nn.Dropout(entity_emb_dropout)
    def forward(self, words, chars, entities=None):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # entities = [sentence_length, batch_size]
        # tags = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim] 
        
        embedding_out = self.word_emb_dropout(self.word_emb(words))
        if not self.use_char_emb:
            return embedding_out
        char_emb_out = self.char_emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels, device=self.device)
        for sent_i in range(sent_len):
            sent_char_emb = char_emb_out[:, sent_i,:,:]
            sent_char_emb_p = sent_char_emb.permute(0,2,1)
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
        char_cnn = self.char_cnn_dropout(char_cnn_max_out)
        char_cnn_p = char_cnn.permute(1,0,2)
        word_features = torch.cat((embedding_out, char_cnn_p), dim=2)
        if self.entity_emb_size != None:
            entity_out = self.entity_emb_dropout(self.entity_emb(entities))
        word_features = torch.cat((word_features, entity_out), dim=2)
        return word_features

class EventEmbedding(nn.Module):

    def __init__(self,
                 event_emb_size=None,
                 event_emb_dim=0,
                 event_emb_dropout=0,
                 event_pad_idx,
                 device
                 ):
        self.device = device

        self.event_emb = nn.Embedding(
            num_embeddings=event_emb_size,
            out_channels=event_emb_dim
        )
        self.event_emb_dropout = nn.Dropout(event_emb_dropout)
        self.output_dim = event_emb_dim
    def forward(self, events):
        return self.event_emb_dropout(self.event_emb(events))