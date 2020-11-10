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

class CNN_Arg(nn.Module):
    def __init__(self,
                 input_dim,
                 cnn_out_channel,
                 cnn_kernels,
                 cnn_dropout,
                 pos_emb_size,
                 pos_emb_dim
                ):
        super().__init__()
        self.pos_emb = nn.Embedding(
            num_embeddings=pos_emb_size,
            embedding_dim=pos_emb_dim
        ) 

        self.entity_emb = nn.Embedding(
            num_embeddings=entity_emb_size,
            embedding_dim=entity_emb_dim
        ) if entity_emb_size is not None else None

        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1,
                out_channels = cnn_out_channel,
                kernel_size = (k, input_dim+pos_emb_dim*2)
            ) for k in cnn_kernels]
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)

    def get_sentence_positional_features(self, batch_size, sentence_length):
        positions = [[abs(j) for j in range(-i, sentence_length-i)] for i in range(sentence_length)]
        positions = [torch.cuda.LongTensor(position) for position in positions]
        positions = [torch.cat([position]*batch_size).resize_(batch_size, position.size(0))
                        for position in positions]
        return positions
    def get_sentence_event_positional_features(self, batch_size, sentence_length, trigger_indexes):
        positions = [[abs(j) for j in range(-i, sentence_length-i)] for i in range(sentence_length)]
        positions = [torch.cuda.LongTensor(position) for position in positions]
        # positions = [torch.cat([position]*batch_size).resize_(batch_size, position.size(0))
        #                 for position in positions]
        positions = torch.cat([positions[i] for i in trigger_indexes]).resize_(batch_size, sentence_length)
        return positions

    def forward(self, words, word_features, trigger_indexes=[2]*128):
        positional_sequences = self.get_sentence_positional_features(words.shape[1], words.shape[0])
        positional_event = self.get_sentence_event_positional_features(words.shape[1], words.shape[0], trigger_indexes)
        
        cnn_input = word_features.permute(1,0,2)
        cnn_out=[]
        for i in range(words.shape[0]):
            x = torch.cat([cnn_input, self.pos_emb(positional_sequences[i]), self.pos_emb(positional_event[])], 2) 
            x = x.unsqueeze(1)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            cnn_out.append(x)
        cnn_out = torch.stack(cnn_out, dim=1)
        cnn_out = cnn_out.permute(1, 0, 2)
        return cnn_out

class LSTMAttn(nn.Module):

    def __init__(self,
                 input_dim,
                 lstm_hidden_dim,
                 lstm_layers,
                 lstm_dropout,
                 word_pad_idx,
                 attn_heads=None,
                 attn_dropout=None
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        #biLSTM layer
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = lstm_hidden_dim,
            num_layers = lstm_layers,
            bidirectional = True,
            dropout = lstm_dropout if lstm_layers > 1 else 0
        )
        #attention layer
        self.attn_heads = attn_heads
        if self.attn_heads:
            self.attn = nn.MultiheadAttention(
                embed_dim = lstm_hidden_dim*2,
                num_heads=attn_heads,
                dropout=attn_dropout
            )
    
    def forward(self, words, word_features):
        lstm_out, _ = self.lstm(word_features)
        if not self.attn_heads:
            return lstm_out
        key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1,0)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        return attn_out

class Fully_Connected(nn.Module):
    def __init__(self,
                 input_dim,
                 fc_dropout,
                 argument_names):
        super().__init__()
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(input_dim, len(argument_names))
    
    def forward(self, words, word_features, tags):
        fc_out = self.fc(self.fc_dropout(word_features))
        return fc_out, "no_loss"