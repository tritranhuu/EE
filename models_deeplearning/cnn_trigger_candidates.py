import os
import time
import gensim
import math
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from sklearn.metrics import classification_report

class CNN(nn.Module):

  def __init__(self, 
                input_dim, 
                embedding_dim,
                max_sent_length,
                pos_emb_dim,  
                cnn_kernels,
                cnn_in_chanel,
                cnn_out_chanel, 
                char_emb_dim,
                char_input_dim,
                char_cnn_filter_num,
                char_cnn_kernel_size,
                output_dim,
                emb_dropout,
                char_cnn_dropout, 
                cnn_dropout, 
                fc_dropout, 
                word_pad_idx,
                char_pad_idx,
                args):
    super().__init__()
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
    self.pos_emb_dim = args.pos_emb_dim
    self.pos_emb = nn.Embedding(
        num_embeddings=args.max_sent_length,
        embedding_dim=args.pos_emb_dim
    )


    self.emb_to_cnn_input = self.get_linear_layer(args.embedding_dim + args.char_emb_dim*args.char_cnn_filter_num,
                                                  args.cnn_out_chanel)


    self.char_cnn_dropout = nn.Dropout(args.char_cnn_dropout)
    # LAYER 2: CNN
    self.convs = nn.ModuleList(
        [nn.Conv2d(
            in_channels = 1,
            out_channels= args.cnn_out_chanel,
            kernel_size = k,

            # padding = /(k-1)//2,
            # padding_mode = 'replicate'
        ) for k in args.cnn_kernels]
    )

    self.cnn_dropout = nn.Dropout(args.cnn_dropout)
    # LAYER 3: Fully-connected

    self.fc_dropout = nn.Dropout(args.fc_dropout)
    self.fc = self.get_linear_layer(args.cnn_out_chanel*3, args.output_dim)
    # self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional
    
  def get_sentence_positional_feature(self, batch_size, sentence_length):
    positions = [[abs(j) for j in range(-i, sentence_length-i)] for i in range(sentence_length)]
    positions = [torch.LongTensor(position) for position in positions]
    positions = [torch.cat([position]*batch_size).resize_(batch_size, position.size(0))
                  for position in positions]
    return positions

  def forward(self, words, chars):
    # words = [sentence length, batch size]
    # chars = [batch size, sentence length, word length)
    # embedding_out = [sentence length, batch size, embedding dim]
    # print(chars.shape)
    embedding_out = self.emb_dropout(self.embedding(words))
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
    
    # cnn_input = self.emb_to_cnn_input(self.fc_dropout(word_features))
    positional_sequences = self.get_sentence_positional_feature(words.shape[1], words.shape[0])
    
    # cnn_input = (batch, sentence_length, embedding_dim)
    cnn_input = word_features.permute(1,0,2)
    cnn_out = []
    for i in range(words.shape[0]):
      x = torch.cat([cnn_input, self.pos_emb(positional_sequences[i])], 2)
      x = self.emb_to_cnn_input(self.fc_dropout(x))

      x = x.unsqueeze(1)
      x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
      x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
      x = torch.cat(x, 1)
      cnn_out.append(x)
    cnn_out = torch.stack(cnn_out, dim=1) # cnn_out = (batch, sentlength, dim)

    
    # cnn_out = torch.cat(cnn_out, dim=2 )
    cnn_out = cnn_out.permute(1,0,2)
    # print(cnn_out.shape)

    # fc_input = (sent_length, batch, out_channel)
    # ner_out = [sentence length, batch size, output dim]
    ner_out = self.fc(self.fc_dropout(cnn_out))
    return ner_out

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

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)