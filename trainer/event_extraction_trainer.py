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

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

class Trainer(object):

    def __init__(self, model_event, model_arg, data, optimizer_cls, loss_fn_cls, device):
        self.device = device

        self.model_event = model_event.to(self.device)
        self.model_arg = model_arg.to(self.device)

        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.argument_pad_idx)

    def get_trigger_pos(self, event_matrix):
        x = event_matrix.argmax(dim=2).permute(1, 0)
        x = [(t>1).nonzero().reshape(-1) for t in x]
        s = []
        for i in range(len(x)):
            a = x[i]
            if x[i].shape[0] <=1:
              s.append(x[i])
            else:
              temp = (torch.cat([torch.cuda.FloatTensor([0]), (a[1:] - a[:-1])])!=1).nonzero().reshape(-1)
              temp = x[i][temp]
              s.append(temp)
        pos = []
        i=0
        while i<len(s):
            if len(s[i]) == 0:
              pos.append(torch.cuda.FloatTensor([-1]))
            elif len(s[i]) == 1:
              pos.append(s[i])
            else:
              a = s[i].reshape(-1,1)
              pos.append(a[0])
              for j in range(1,len(s)-i):
                if s[i+j].shape[0] == s[i].shape[0] and torch.allclose(s[i+j],s[i]):
                    if j<a.shape[0]:
                      pos.append(a[j])
                    else:
                      pos.append(a[-1])                    
                else:
                  break
              i+=(j-1)
            i+=1
        pos = torch.cat(pos)
        return pos[:len(s)]

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        idx2tag = self.data.argument_field.vocab.itos
        self.model_event.eval()
        self.model_srg.eval()
        with torch.no_grad():
          # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)
                # if words.shape[0] < 5:
                #   continue
                chars = batch.char.to(self.device)
                entities = batch.entity.to(self.device)
                events = batch.event.to(self.device)
                true_tags = batch.argument.to(self.device)
                pred_event_tags, _ = self.model_event(words, chars)
                
                trigger_indexes = self.get_trigger_pos(pred_event_tags)

                # pred_event_tags = pred_event_tags.view(-1, pred_event_tags.shape[-1])
                pred_tags, _ = self.model_arg(words, chars, entities, pred_event_tags.argmax(dim=2), trigger_index)

                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                
                pred_tags_epoch.extend([idx2tag[i] for i in pred_tags.argmax(dim=1).cpu().numpy()])
                true_tags_epoch.extend([idx2tag[i] for i in true_tags.cpu().numpy()])
        epoch_score = f1_score(true_tags_epoch, pred_tags_epoch, average="micro",labels=list(self.data.argument_field.vocab.stoi.keys())[2:])
        epoch_p1 = precision_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.argument_field.vocab.stoi.keys())[2:])
        epoch_r1 = recall_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.argument_field.vocab.stoi.keys())[2:])
        print(classification_report(true_tags_epoch,pred_tags_epoch,labels=list(self.data.argument_field.vocab.stoi.keys())[2:]))
        return epoch_loss / len(self.data.train_iter), epoch_score, epoch_p1, epoch_r1

    def test(self):
        train, train_f1, train_p1, train_r1 = self.evaluate(self.data.val_iter)
        print(f"\tTrn Loss: {train_loss:.3f} | Trn P1: {train_p1 * 100:.2f}%  R1: {train_r1 * 100:.2f}% F1: {train_f1 * 100:.2f}%")
        val_loss, val_f1, val_p1, val_r1 = self.evaluate(self.data.val_iter)
        print(f"\tVal Loss: {val_loss:.3f} | Val P1: {val_p1 * 100:.2f}%  R1: {val_r1 * 100:.2f}% F1: {val_f1 * 100:.2f}%")
        test_loss, test_f1, test_p1, test_r1 = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test P1: {test_p1 * 100:.2f}%  R1: {test_r1 * 100:.2f}% F1: {test_f1 * 100:.2f}%")

