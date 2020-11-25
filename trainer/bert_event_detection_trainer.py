import os
import time
import gensim
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchtext.data import Field, BucketIterator, NestedField
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

class Trainer(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls, device, tokenizer, checkpoint_path=None):
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.event_pad_idx)
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def get_trigger_pos(self, event_matrix):
        x = event_matrix.permute(1,0)
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
        # print(pos.shape, pos, s, len(s))
        return pos[:len(s)]

    def get_bert_tokens (self, words):
        batch_iter = []
        idx2words = self.data.word_field.vocab.itos
        t = [[idx2words[w] for w in s if w!=1] for s in words.T.numpy()]
        for sent in t:
            len_words = []
            tokens = []
            for word in sent:
                subwords = self.tokenizer.tokenize(word)
                tokens.append(subwords)
                len_words.append(len(subwords))
            n_words = len(np.where(np.cumsum(len_words) <= (self.tokenizer.max_len - 2))[0])
            tokens = [self.tokenizer.cls_token, *(sum(tokens[:n_words], [])), self.tokenizer.sep_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            batch_iter.append((token_ids, n_words, torch.tensor(len_words)))
        max_length = np.max([x[0].size() for x in batch_iter])
        batch_tokens = torch.ones(len(batch_iter), max_length).long()
        batch_len_words = torch.ones(len(batch_iter), max_length).long()
        for i, c in enumerate(batch_iter):
            batch_tokens[i][:c[0].size(0)] = c[0]
            batch_len_words[i][:c[2].size(0)] = c[2]
        _, n_words, _ = zip(*batch_iter)
        batch_n_words = torch.tensor(list(n_words))
        return batch_tokens, batch_n_words, batch_len_words

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        idx2tag = self.data.event_field.vocab.itos
        self.model.train()
        for batch in self.data.train_iter:
        # words = [sent len, batch size]
            words = batch.word.to(self.device)
            tokens = self.get_bert_tokens(words)
        # tags = [sent len, batch size]
            true_tags = batch.event.to(self.device)
            self.optimizer.zero_grad()
            pred_tags, _ = self.model(tokens)
            # self.get_trigger_pos(true_tags)
        # to calculate the loss and accuracy, we flatten both prediction and true tags
        # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
        # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)
            batch_loss = self.loss_fn(pred_tags, true_tags)
           
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            
            pred_tags_epoch.extend([idx2tag[i] for i in pred_tags.argmax(dim=1).cpu().numpy()])
            true_tags_epoch.extend([idx2tag[i] for i in true_tags.cpu().numpy()])
        # epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch)
        epoch_score = f1_score(true_tags_epoch, pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        epoch_p1 = precision_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        epoch_r1 = recall_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        return epoch_loss / len(self.data.train_iter), epoch_score, epoch_p1, epoch_r1

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        idx2tag = self.data.event_field.vocab.itos
        self.model.eval()
        with torch.no_grad():
          # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)
                if words.shape[0] < 5:
                  continue
                tokens = self.get_bert_tokens(words)
                true_tags = batch.event.to(self.device)
                pred_tags, _ = self.model(tokens)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                
                pred_tags_epoch.extend([idx2tag[i] for i in pred_tags.argmax(dim=1).cpu().numpy()])
                true_tags_epoch.extend([idx2tag[i] for i in true_tags.cpu().numpy()])
        epoch_score = f1_score(true_tags_epoch, pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        epoch_p1 = precision_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        epoch_r1 = recall_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        # print(classification_report(true_tags_epoch,pred_tags_epoch, labels=list(self.data.event_field.vocab.stoi.keys())[2:]))
        return epoch_loss / len(self.data.train_iter), epoch_score, epoch_p1, epoch_r1

  # main training sequence
    def train_live(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_f1, train_p1, train_r1 = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn P1: {train_p1 * 100:.2f}%  R1: {train_r1 * 100:.2f}% F1: {train_f1 * 100:.2f}%")
            val_loss, val_f1, val_p1, val_r1 = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val P1: {val_p1 * 100:.2f}%  R1: {val_r1 * 100:.2f}% F1: {val_f1 * 100:.2f}%")
        test_loss, test_f1, test_p1, test_r1 = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test P1: {test_p1 * 100:.2f}%  R1: {test_r1 * 100:.2f}% F1: {test_f1 * 100:.2f}%")
    
    def train(self, max_epochs=20, no_improvement=None):
        history = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
        }
        elapsed_train_time = 0
        best_val_f1 = 0
        best_epoch = None
        lr_scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer,
            patience=3,
            factor=0.3,
            mode='max',
            verbose=True
        )
        epoch = 1
        n_stagnant = 0
        stop = False
        while not stop:
            start_time = time.time()
            train_loss, train_f1, train_p1, train_r1 = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            val_loss, val_f1, val_p1, val_r1 = self.evaluate(self.data.val_iter)
            lr_scheduler.step(val_f1)
            if self.checkpoint_path and val_f1 > (1.01*best_val_f1):
                print(f"Epoch {epoch:5d}: found better Val F1: {val_f1:.4f} (Train F1: {train_f1:.4f}), saving model...")
                self.model.save_state(self.checkpoint_path)
                best_val_f1 = val_f1
                best_epoch = epoch
                n_stagnant = 0
            else:
                n_stagnant += 1
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            if epoch >= max_epochs:
                print(f"Reach maximum number of epoch: {epoch}, stop training.")
                stop = True
            elif no_improvement is not None and n_stagnant >= no_improvement:
                print(f"No improvement after {n_stagnant} epochs, stop training.")
                stop = True
            else:
                epoch += 1
        if self.checkpoint_path and best_val_f1 > 0:
            self.model.load_state(self.checkpoint_path)
        test_loss, test_f1, test_p1, test_r1 = self.evaluate(self.data.test_iter)
        history["best_val_f1"] = best_val_f1
        history["best_epoch"] = best_epoch
        history["test_loss"] = test_loss
        history["test_f1"] = test_f1
        history["elapsed_train_time"] = elapsed_train_time
        return history
