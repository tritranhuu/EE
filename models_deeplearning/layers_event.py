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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze().transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNSequence(nn.Module):
    def __init__(self,
                 input_dim,
                 cnn_out_channel,
                 cnn_kernels,
                 cnn_dropout
                 ):
        super().__init__()
        self.emb_to_cnn_input = nn.Linear(
            in_features = input_dim,
            out_features=cnn_out_channel
        )
        #CNN layer
        self.convs = nn.ModuleList(
            [nn.Conv1d(
                in_channels = cnn_out_channel,
                out_channels= cnn_out_channel*2,
                kernel_size = k,
                padding = (k-1)//2,
                # padding_mode = 'replicate'
            ) for k in cnn_kernels]
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.scale = torch.sqrt(torch.cuda.FloatTensor([0.5]))
    def forward(self, words, word_features):
        cnn_input = self.emb_to_cnn_input(self.cnn_dropout(word_features))
        # word_features = (batch, embedding_dim, sentence_length)

        cnn_input = cnn_input.permute(1,2,0)
    
        for i, conv in enumerate(self.convs):
            # conved = (batch, cnn_out_chanel, sentlength)
            conved = conv(self.cnn_dropout(cnn_input))
            conved = nn.functional.glu(conved, dim=1)
            conved = (conved+cnn_input)*self.scale
            cnn_input = conved

        cnn_out = conved.permute(2,0,1)
        return cnn_out

class CNN_TC(nn.Module):
    def __init__(self,
                 input_dim,
                 cnn_out_channel,
                 cnn_kernels,
                 cnn_dropout,
                 pos_emb_size=None,
                 pos_emb_dim=0
                ):
        super().__init__()
        self.pos_emb = nn.Embedding(
            num_embeddings=pos_emb_size,
            embedding_dim=pos_emb_dim
        ) if pos_emb_size is not None else None
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels = 1,
                out_channels = cnn_out_channel,
                kernel_size = (k, input_dim+pos_emb_dim)
            ) for k in cnn_kernels]
        )

        self.cnn_dropout = nn.Dropout(cnn_dropout)
    def get_sentence_positional_features(self, batch_size, sentence_length):
        positions = [[abs(j) for j in range(-i, sentence_length-i)] for i in range(sentence_length)]
        positions = [torch.cuda.LongTensor(position) for position in positions]
        positions = [torch.cat([position]*batch_size).resize_(batch_size, position.size(0))
                        for position in positions]
        return positions
 
    def forward(self, words, word_features):
        
        positional_sequences = self.get_sentence_positional_features(words.shape[1], words.shape[0])
        cnn_input = word_features.permute(1,0,2)
        cnn_out=[]
        for i in range(words.shape[0]):
            x = torch.cat([cnn_input, self.pos_emb(positional_sequences[i])], 2) if self.pos_emb is not None  else cnn_input
            x = x.unsqueeze(1)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            cnn_out.append(x)
        cnn_out = torch.stack(cnn_out, dim=1)
        cnn_out = cnn_out.permute(1, 0, 2)
        return cnn_out

class Transformer(nn.Module):

    def __init__(self,
                 input_dim,
                 attn_heads,
                 attn_dropout,
                 trf_layers,
                 fc_hidden,
                 word_pad_idx
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        self.position_encoder = PositionalEncoding(
            d_model = input_dim
        )
        layers = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=attn_heads,
            activation='relu',
            dropout=attn_dropout
        )
        self.trf = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers=args.trf_layers
        )
        self.fc = nn.Linear(
            in_features=input_dim,
            out_features=fc_hidden
        )
        self.fc_gelu = nn.GELU()
        self.fc_norm = nn.LayerNorm(fc_hidden)
        self.output_dim = fc_hidden

    def forward(self, words, word_features):
        key_padding_mask = torch.as_tensor(words==self.word_pad_idx).permute(1,0)
        pos_out = self.position_encoder(word_features)
        trf_out = self.trf(pos_out, src_key_padding_mask=key_padding_mask)
        fc_out = self.fc_norm(self.fc_gelu(self.fc(trf_out)))
        return fc_out

class CRF_(nn.Module):

    def __init__(self,
                 input_dim,
                 fc_dropout,
                 word_pad_idx,
                 tag_names
                 ):
        super().__init__()
        self.word_pad_idx = word_pad_idx
        #Fully connected layer
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(input_dim, len(tag_names))
        #CRF
        self.crf = CRF(num_tags=len(tag_names))
    
    def forward(self, words, word_features, tags):
        # fc_out = [sentence length, batch size, output dim]
        fc_out = self.fc(self.fc_dropout(word_features))
        crf_mask = words != self.word_pad_idx
        crf_out = self.crf.decode(fc_out, mask=crf_mask)
        crf_loss = -self.crf(fc_out, tags=tags, mask=crf_mask) if tags is not None else None
        return crf_out, crf_loss

class Fully_Connected(nn.Module):
    def __init__(self,
                 input_dim,
                 fc_dropout,
                 tag_names):
        super().__init__()
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(input_dim, len(tag_names))
    
    def forward(self, words, word_features, tags):
        fc_out = self.fc(self.fc_dropout(word_features))
        return fc_out, "no_loss"


