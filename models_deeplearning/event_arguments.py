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

from embedding.embedding import Embeddings, EventEmbedding
from models_deeplearning.layers_argument import *

class Model_EA(nn.Module):
    def __init__(self,
                 word_input_dim,
                 word_pad_idx,
                 char_pad_idx,
                 entity_pad_idx,
                 event_pad_idx,
                 argument_names,
                 device,
                 model_arch="bilstm",
                 word_emb_dim=300,
                 word_emb_pretrained=None,
                 word_emb_dropout=0.5,
                 word_emb_froze=False,
                 use_char_emb=False,
                 pos_emb_size=None,
                 pos_emb_dim=0,
                 entity_emb_size=None,
                 entity_emb_dim=0,
                 entity_emb_dropout=0,
                 char_input_dim=None,
                 char_emb_dim=None,
                 char_emb_dropout=None,
                 char_cnn_filter_num=None,
                 char_cnn_kernel_size=None,
                 char_cnn_dropout=None,
                 event_emb_size=None,
                 event_emb_dim=0,
                 event_emb_dropout=0,
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
        self.events_embeddings = EventEmbedding(
            event_emb_dim=event_emb_dim,
            event_emb_size=event_emb_size,
            event_pad_idx=event_pad_idx,
            event_emb_dropout=event_emb_dropout,
            device=device
        )
        if model_arch.lower() == "cnn_trig":
            # CNN for Trigger Candidates
            self.encoder = CNN_Arg(
                input_dim=self.embeddings.output_dim,
                cnn_out_channel=cnn_out_channel,
                cnn_kernels=cnn_kernels,
                cnn_dropout=cnn_dropout,
                entity_emb_size=entity_emb_size,
                entity_emb_dim=entity_emb_dim,
                pos_emb_size=pos_emb_size,
                pos_emb_dim=pos_emb_dim
            )
            encoder_output_dim = cnn_out_channel*len(cnn_kernels)
        elif model_arch.lower() == "bilstm":
            # CNN for Trigger Candidates
            self.encoder = LSTMAttn(
                 input_dim=self.embeddings.output_dim + self.events_embeddings.output_dim,
                 lstm_hidden_dim=lstm_hidden_dim,
                 lstm_layers=lstm_layers,
                 lstm_dropout=lstm_dropout,
                 word_pad_idx=word_pad_idx,
                 attn_heads=attn_heads,
                 attn_dropout=attn_dropout
            )
            encoder_output_dim = lstm_hidden_dim * 2
        else:
            raise ValueError("param `model_arch` unknown")
        # CRF
        # if use_crf:
        #     self.final_layer = CRF_(
        #         input_dim=encoder_output_dim,
        #         fc_dropout=fc_dropout,
        #         word_pad_idx=word_pad_idx,
        #         tag_names=tag_names
        #     )
        # else:
        self.final_layer = Fully_Connected(
            input_dim=encoder_output_dim,
            fc_dropout=fc_dropout,
            argument_names=argument_names
        )

    def forward(self, words, chars, entities, events, tags=None):
        word_features = self.embeddings(words, chars, entities)
        event_features = self.events_embeddings(events)
        features = torch.cat((word_features, event_features), dim=2)
        encoder_out = self.encoder(words, word_features)
        # fc_out = [sentence length, batch size, output dim]
        ea_out, ea_loss = self.final_layer(words, encoder_out, tags)
        return ea_out, ea_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)