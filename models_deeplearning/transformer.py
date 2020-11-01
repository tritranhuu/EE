import os
import time
import gensim
import math
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):

  def __init__(self, 
                # input_dim, 
                # embedding_dim, 
                # hidden_dim, 
                # char_emb_dim,
                # char_input_dim,
                # char_cnn_filter_num,
                # char_cnn_kernel_size,
                # output_dim, 
                # lstm_layers,
                # attn_heads,
                # emb_dropout,
                # cnn_dropout, 
                # lstm_dropout,
                # attn_dropout, 
                # fc_dropout, 
                # word_pad_idx,
                # char_pad_idx,
                # tag_pad_idx,
                args,
                device):
    super().__init__()
    self.char_pad_idx = args.char_pad_idx
    self.word_pad_idx = args.word_pad_idx
    self.tag_pad_idx = args.tag_pad_idx
    self.device = device
    self.embedding_dim = args.embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=args.input_dim, 
        embedding_dim=args.embedding_dim, 
        padding_idx=args.word_pad_idx
    )
    self.emb_dropout = nn.Dropout(args.emb_dropout)
    # LAYER 1B: Char Embedding-CNN
    self.char_emb_dim = args.char_emb_dim
    self.char_emb = nn.Embedding(
        num_embeddings=args.char_input_dim,
        embedding_dim=args.char_emb_dim,
        padding_idx=args.char_pad_idx
    )
    self.char_cnn = nn.Conv1d(
        in_channels=args.char_emb_dim,
        out_channels=args.char_emb_dim * args.char_cnn_filter_num,
        kernel_size=args.char_cnn_kernel_size,
        groups=args.char_emb_dim  # different 1d conv for each embedding dim
    )
    self.cnn_dropout = nn.Dropout(args.cnn_dropout)
    # LAYER 2: Transformer
    all_emb_size = args.embedding_dim + (args.char_emb_dim*args.char_cnn_filter_num)
    self.position_encoder = PositionalEncoding(
        d_model = all_emb_size
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=all_emb_size,
        nhead=args.attn_heads,
        activation='relu',
        dropout=args.trf_dropout
    )
    self.encoder = nn.TransformerEncoder(
        encoder_layer = encoder_layer,
        num_layers=args.trf_layers
    )

    # LAYER 3: 2-layers fully-connected with GELU activation in-between
    self.fc1 = self.get_linear_layer(
        input_dim=all_emb_size,
        output_dim=args.fc_hidden
    )
    self.fc1_gelu = nn.GELU()
    self.fc1_norm = nn.LayerNorm(args.fc_hidden)
    self.fc2_dropout = nn.Dropout(args.fc_dropout)
    self.fc2 = self.get_linear_layer(
        input_dim=args.fc_hidden,
        output_dim=args.output_dim
    )
    # LAYER 4: CRF
    self.tag_pad_idx = args.tag_pad_idx
    self.crf = CRF(num_tags=args.output_dim)
    # for name, param in self.named_parameters


  def forward(self, words, chars, tags=None):
    # words = [sentence length, batch size]
    # chars = [batch size, sentence length, word length)
    # embedding_out = [sentence length, batch size, embedding dim]
    embedding_out = self.emb_dropout(self.embedding(words)).to(self.device)
    # lstm_out = [sentence length, batch size, hidden dim * 2]
    char_emb_out = self.emb_dropout(self.char_emb(chars)).to(self.device)
    batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
    char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
        # for character embedding, we need to iterate over sentences
    for sent_i in range(sent_len):
      sent_char_emb = char_emb_out[:, sent_i,:,:]
      sent_char_emb_p = sent_char_emb.permute(0,2,1)
      char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
      char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
    char_cnn = self.cnn_dropout(char_cnn_max_out)
    char_cnn_p = char_cnn.permute(1,0,2).to(self.device)
    word_features = torch.cat((embedding_out, char_cnn_p), dim=2)

    #transformer forward
    key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1, 0)
    pos_out = self.position_encoder(word_features)
    enc_out = self.encoder(pos_out, src_key_padding_mask=key_padding_mask)
    fc1_out = self.fc1_norm(self.fc1_gelu(self.fc1(enc_out)))
    fc2_out = self.fc2(self.fc2_dropout(fc1_out))

    crf_mask = words != self.word_pad_idx
    crf_out = self.crf.decode(fc2_out, mask=crf_mask)
    crf_loss = -self.crf(fc2_out, tags=tags, mask=crf_mask) if tags is not None else None
    return crf_out, crf_loss
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
            

        