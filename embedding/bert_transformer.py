import torch
import torch.nn as nn
import numpy as np


class BertTransformer(nn.Module):
    def __init__(self, bert_model, device):
        super().__init__()
        self.bert_model = bert_model
        self.device = device

        self.output_dim = 768
        
    # def remove_cls_sep_tokens(self, hidden, attention_mask):
    #     n_tokens = attention_mask.sum(dim=-1)
    #     attention_mask[range(n_tokens.size(0)), n_tokens.long()-2] = 0
    #     expand_attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden)
    #     return (expand_attention_mask.to(self.device) * hidden)[:, 1:, :]
    
    # def reduction_stack(self, x: torch.Tensor, len_words: torch.Tensor, num_words):
    #     max_len_word = len_words.max().item()
    #     hidden_size = x.size(-1)
    #     if max_len_word == 1:
    #         return x[:num_words]
        
    #     idx = []
    #     len_words_cumsum = [0, *torch.cumsum(len_words, dim=0, dtype=torch.long).detach().cpu().tolist()]
    #     num_pad_tokens = (max_len_word - len_words).detach().cpu().tolist()
    #     for i in range(num_words):
    #         idx.extend(range(len_words_cumsum[i], len_words_cumsum[i+1]))
    #         idx.extend([0]*num_pad_tokens[i])
        
    #     x = x[idx].view(num_words, max_len_word, hidden_size).sum(dim=1)
    #     return x
    # def token_reduction(self, x, n_words, len_words):
    #     batch_size, _, hidden_size = x.size()
    #     output = torch.zeros(size=(batch_size, n_words.max().item(), hidden_size), dtype=x.dtype, device=x.device)
    #     mask = torch.zeros(size=(batch_size, n_words.max().item()), dtype=torch.bool, device=x.device)
    #     for i in range(batch_size):
    #         output[i, :n_words[i]] = self.reduction_stack(x[i], len_words[i], num_words=n_words[i])
    #         mask[i, :n_words[i]] = 1
    #         return output, mask
    def token_reduction(self, x, n_words, word_mask):
      batch_size, _, hidden_size = x.size()
      max_sent_length = n_words.max().item()
      output = torch.zeros(size=(batch_size, max_sent_length, hidden_size), dtype=x.dtype, device=x.device)
      for i in range(batch_size):
        mask = word_mask[i].nonzero().reshape(-1)[:max_sent_length]
        output[i][:mask.size(0)] = x[i][mask]
      return output 

    def forward(self, token_ids, n_words, word_mask):
        attention_mask = token_ids != 1
        hidden, _ = self.bert_model(input_ids=token_ids, attention_mask=attention_mask.to(self.device))
        reduction = hidden
        # output = self.token_reduction(reduction, n_words, word_mask)
        # reduction = self.remove_cls_sep_tokens(hidden[0], attention_mask)
        # output, _ = self.token_reduction(reduction, n_words=n_words, len_words=len_words)
        output = reduction
        return output.permute(1,0,2)
