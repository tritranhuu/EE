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
                 word_pad_idx,
                 char_pad_idx,
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
        
    def forward(self, words, chars):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
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
        return word_features
        
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

class Model(nn.Module):
    def __init__(self,
                 word_input_dim,
                 word_pad_idx,
                 char_pad_idx,
                 tag_names,
                 device,
                 model_arch="bilstm",
                 word_emb_dim=300,
                 word_emb_pretrained=None,
                 word_emb_dropout=0.5,
                 word_emb_froze=False,
                 use_char_emb=False,
                 pos_emb_size=None,
                 pos_emb_dim=0,
                 char_input_dim=None,
                 char_emb_dim=None,
                 char_emb_dropout=None,
                 char_cnn_filter_num=None,
                 char_cnn_kernel_size=None,
                 char_cnn_dropout=None,
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 lstm_dropout=0.1,
                 attn_heads=None,
                 attn_dropout=None,
                 trf_layers=None,
                 cnn_out_channel=None,
                 cnn_kernels=None,
                 cnn_dropout=None,
                 fc_hidden=None,
                 fc_dropout=0.25,
                 use_crf=False):
        super().__init__()
        self.embeddings = Embeddings(
            word_input_dim=word_input_dim,
            word_emb_dim=word_emb_dim,
            word_emb_pretrained=word_emb_pretrained,
            word_emb_dropout=word_emb_dropout,
            word_emb_froze=word_emb_froze,
            use_char_emb=use_char_emb,
            char_input_dim=char_input_dim,
            char_emb_dim=char_emb_dim,
            char_emb_dropout=char_emb_dropout,
            char_cnn_filter_num=char_cnn_filter_num,
            char_cnn_kernel_size=char_cnn_kernel_size,
            char_cnn_dropout=char_cnn_dropout,
            word_pad_idx=word_pad_idx,
            char_pad_idx=char_pad_idx,
            device=device
        )
        if model_arch.lower() == "bilstm":
            # LSTM-Attention
            self.encoder = LSTMAttn(
                 input_dim=self.embeddings.output_dim,
                 lstm_hidden_dim=lstm_hidden_dim,
                 lstm_layers=lstm_layers,
                 lstm_dropout=lstm_dropout,
                 word_pad_idx=word_pad_idx,
                 attn_heads=attn_heads,
                 attn_dropout=attn_dropout
            )
            encoder_output_dim = lstm_hidden_dim * 2
        elif model_arch.lower() == "transformer":
            # Transformer
            self.encoder = Transformer(
                input_dim=self.embeddings.output_dim,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
                trf_layers=trf_layers,
                fc_hidden=fc_hidden,
                word_pad_idx=word_pad_idx
            )
            encoder_output_dim = self.encoder.output_dim   
        elif model_arch.lower() == "cnn_seq":
            # Transformer
            self.encoder = CNNSequence(
                input_dim=self.embeddings.output_dim,
                cnn_out_channel=cnn_out_channel,
                cnn_kernels=cnn_kernels,
                cnn_dropout=cnn_dropout
            )
            encoder_output_dim = cnn_out_channel
        elif model_arch.lower() == "cnn_trig":
            # CNN for Trigger Candidates
            self.encoder = CNN_TC(
                input_dim=self.embeddings.output_dim,
                cnn_out_channel=cnn_out_channel,
                cnn_kernels=cnn_kernels,
                cnn_dropout=cnn_dropout,
                pos_emb_size=pos_emb_size,
                pos_emb_dim=pos_emb_dim
            )
            encoder_output_dim = cnn_out_channel*len(cnn_kernels)
        else:
            raise ValueError("param `model_arch` unknown")
        # CRF
        if use_crf:
            self.final_layer = CRF_(
                input_dim=encoder_output_dim,
                fc_dropout=fc_dropout,
                word_pad_idx=word_pad_idx,
                tag_names=tag_names
            )
        else:
            self.final_layer = Fully_Connected(
                input_dim=encoder_output_dim,
                fc_dropout=fc_dropout,
                tag_names=tag_names
            )

    def forward(self, words, chars, tags=None):
        word_features = self.embeddings(words, chars)
        encoder_out = self.encoder(words, word_features)
        # fc_out = [sentence length, batch size, output dim]
        ed_out, ed_loss = self.final_layer(words, encoder_out, tags)
        return ed_out, ed_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)