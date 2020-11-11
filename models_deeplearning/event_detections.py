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

from embedding.embedding import Embeddings
from models_deeplearning.layers_event import *

class Model_ED(nn.Module):
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
    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_state(self, path):
        self.load_state_dict(torch.load(path))
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)