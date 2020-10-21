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
from sklearn.metrics import classification_report, f1_score

class ED(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls):
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != 0).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        # print(max_preds[non_pad_elements][1].item())
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

    def f_score(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()
        y_pred = []
        y_true = []
        for i in range(max_preds[non_pad_elements].shape[0]):
            y_pred.append(self.data.tag_field.vocab.itos[max_preds[non_pad_elements][i].item()])
            y_true.append(self.data.tag_field.vocab.itos[y[non_pad_elements][i].item()])
        # print(classification_report(y_pred=np.array(y_pred), y_true=np.array(y_true), labels=list(self.data.tag_field.vocab.stoi.keys())[2:]))
        print(f1_score(y_pred=np.array(y_pred), y_true=np.array(y_true), labels=list(self.data.tag_field.vocab.stoi.keys())[2:], average='micro'))

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.data.train_iter:
        # words = [sent len, batch size]
            words = batch.word
            chars = batch.char
        # tags = [sent len, batch size]
            true_tags = batch.tag
            self.optimizer.zero_grad()
            pred_tags = self.model(words, chars)
        # to calculate the loss and accuracy, we flatten both prediction and true tags
        # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
        # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)
            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)
            # self.f_score(pred_tags, true_tags)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        y_pred = []
        y_true = []
        idx2tag = self.data.tag_field.vocab.itos
        self.model.eval()
        with torch.no_grad():
          # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word
                chars = batch.char
                true_tags = batch.tag
                pred_tags = self.model(words, chars)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
                y_pred.extend([idx2tag[i] for i in pred_tags.argmax(dim=1).numpy()])
                y_true.extend([idx2tag[i] for i in true_tags.numpy()])
        print(classification_report(y_pred=np.array(y_pred), y_true=np.array(y_true), labels=list(self.data.tag_field.vocab.stoi.keys())[2:]))
        # print(f1_score(y_pred=np.array(y_pred), y_true=np.array(y_true), labels=list(self.data.tag_field.vocab.stoi.keys())[2:], average='micro'))

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

  # main training sequence
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = ED.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")
