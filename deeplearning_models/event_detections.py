import os
import time
import gensim
import math
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam

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
            self.word_emb = nn.Embeddings.from_pretrained(
                embeddings=torch.as_tensor(word_emb_pretrained),
                padding_idx = self.word_pad_idx,
                freeze=word_emb_froze
            )
        else:
            self.word_emb = nn.Embeddings(
                num_embeddings=word_input_dim,
                embedding_dim=word_emb_dim,
                padding_idx=self.word_pad_idx
            )
            self.word_emb.weight.data[self.word_pad_idx] = torch.zeros(word_emb_dim)
        self.word_emb_dropout = nn.Dropout(word_emb_dropout)
            
