import os
import time
import gensim
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator, NestedField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab
from torchcrf import CRF

import numpy as np
from sklearn.metrics import classification_report

class BiLSTM_CRF(nn.Module):

  def __init__(self, 
                input_dim, 
                embedding_dim, 
                hidden_dim, 
                char_emb_dim,
                char_input_dim,
                char_cnn_filter_num,
                char_cnn_kernel_size,
                output_dim, 
                lstm_layers,
                attn_heads,
                emb_dropout,
                cnn_dropout, 
                lstm_dropout,
                attn_dropout, 
                fc_dropout, 
                word_pad_idx,
                char_pad_idx,
                tag_pad_idx):
    super().__init__()
    self.char_pad_idx = char_pad_idx
    self.word_pad_idx = word_pad_idx
    self.tag_pad_idx = tag_pad_idx

    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=input_dim, 
        embedding_dim=embedding_dim, 
        padding_idx=word_pad_idx
    )
    self.emb_dropout = nn.Dropout(emb_dropout)
    # LAYER 1B: Char Embedding-CNN
    self.char_emb_dim = char_emb_dim
    self.char_emb = nn.Embedding(
        num_embeddings=char_input_dim,
        embedding_dim=char_emb_dim,
        padding_idx=char_pad_idx
    )
    self.char_cnn = nn.Conv1d(
        in_channels=char_emb_dim,
        out_channels=char_emb_dim * char_cnn_filter_num,
        kernel_size=char_cnn_kernel_size,
        groups=char_emb_dim  # different 1d conv for each embedding dim
    )
    self.cnn_dropout = nn.Dropout(cnn_dropout)
    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=embedding_dim + (char_emb_dim*char_cnn_filter_num),
        hidden_size=hidden_dim,
        num_layers=lstm_layers,
        bidirectional=True,
        dropout=lstm_dropout if lstm_layers > 1 else 0
    )

    # LAYER 3: Self-attention
    self.attn = nn.MultiheadAttention(
      embed_dim=hidden_dim*2,
      num_heads=attn_heads,
      dropout=attn_dropout
    )
    # LAYER 4: Fully-connected
    self.fc_dropout = nn.Dropout(fc_dropout)
    # self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional
    self.fc = self.get_linear_layer(hidden_dim * 2, output_dim)

    # LAYER 4: CRF
    self.tag_pad_idx = tag_pad_idx
    self.crf = CRF(num_tags=output_dim)
    # for name, param in self.named_parameters


  def forward(self, words, chars, tags=None):
    # words = [sentence length, batch size]
    # chars = [batch size, sentence length, word length)
    # embedding_out = [sentence length, batch size, embedding dim]
    embedding_out = self.emb_dropout(self.embedding(words))
    # lstm_out = [sentence length, batch size, hidden dim * 2]
    char_emb_out = self.emb_dropout(self.char_emb(chars))
    batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
    char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
        # for character embedding, we need to iterate over sentences
    for sent_i in range(sent_len):
      sent_char_emb = char_emb_out[:, sent_i,:,:]
      sent_char_emb_p = sent_char_emb.permute(0,2,1)
      char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
      char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
    char_cnn = self.cnn_dropout(char_cnn_max_out)
    char_cnn_p = char_cnn.permute(1,0,2)
    word_features = torch.cat((embedding_out, char_cnn_p), dim=2)

    lstm_out, _ = self.lstm(word_features)
    # ner_out = [sentence length, batch size, output dim]
    key_padding_mask = torch.as_tensor(words==self.word_pad_idx).permute(1,0)
    attn_out, attn_weight = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)

    fc_out = self.fc(self.fc_dropout(attn_out))
    crf_mask = words != self.word_pad_idx
    crf_out = self.crf.decode(fc_out, mask=crf_mask)
    crf_loss = -self.crf(fc_out, tags=tags, mask=crf_mask) if tags is not None else None
    return crf_out, crf_loss, attn_weight
    # if tags is not None:
    #     mask = tags != self.tag_pad_idx
    #     crf_out = self.crf.decode(fc_out, mask=mask)
    #     crf_loss = -self.crf(fc_out, tags=tags, mask=mask, reduction='sum')
    # else:
    #     crf_out = self.crf.decode(fc_out)
    #     crf_loss = None

    return crf_out, crf_loss

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

  def init_embeddings(self, char_pad_idx, word_pad_idx, pretrained=None, freeze=True):
    # initialize embedding for padding as zero
    self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
    self.char_emb.weight.data[char_pad_idx] = torch.zeros(self.char_emb_dim)
    if pretrained is not None:
        self.embedding = nn.Embedding.from_pretrained(
          embeddings=torch.as_tensor(pretrained),
          padding_idx=word_pad_idx,
          freeze=freeze
        )
  @staticmethod
  def get_linear_layer(input_dim, output_dim):
    linear_layer = nn.Linear(input_dim, output_dim, bias=True)
    mean = 0.0  # std_dev = np.sqrt(variance)
    std_dev = np.sqrt(2 / (output_dim + input_dim))  # np.sqrt(1 / m) # np.sqrt(1 / n)
    weight = np.random.normal(mean, std_dev, size=(output_dim, input_dim)).astype(np.float32)
    std_dev = np.sqrt(1 / output_dim)  # np.sqrt(2 / (m + 1))
    bt = np.random.normal(mean, std_dev, size=output_dim).astype(np.float32)

    linear_layer.weight.data = torch.tensor(weight, requires_grad=True)
    linear_layer.bias.data = torch.tensor(bt, requires_grad=True)
    return linear_layer

#   def init_crf_transitions(self, tag_names, imp_value=-100):
#       num_tags = len(tag_names)
#       for i in range(num_tags):



  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
