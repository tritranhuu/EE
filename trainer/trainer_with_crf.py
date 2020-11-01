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

class Trainer(object):

    def __init__(self, model, data, optimizer_cls, loss_fn_cls, device):
        self.device = device
        self.model = model.to(self.device)
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
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        # print(flatten_y)
        correct = [pred==tag for pred, tag in zip(flatten_preds, flatten_y)]
        return sum(correct) / len(correct) if len(correct) > 0 else 0

    def f1_positive(self, preds, y):
        index_o = self.data.tag_field.vocab.stoi["O"]
        # take all labels except padding and "O"
        positive_labels = [i for i in range(len(self.data.tag_field.vocab.itos))
                           if i not in (self.data.tag_pad_idx, index_o)]
        # make the prediction one dimensional to follow sklearn f1 score input param
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        # remove prediction for padding and "O"
        positive_preds = [pred for pred in flatten_preds
                          if pred not in (self.data.tag_pad_idx, index_o)]
        # make the true tags one dimensional to follow sklearn f1 score input param
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        # average "micro" means we take weighted average of the class f1 score
        # weighted based on the number of support
        return f1_score(
            y_true=flatten_y,
            y_pred=flatten_preds,
            labels=positive_labels,
            average="micro"
        ) if len(positive_preds) > 0 else 0

    def epoch(self):
        epoch_loss = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        self.model.train()
        for batch in self.data.train_iter:
        # words = [sent len, batch size]
            words = batch.word.to(self.device)
            chars = batch.char.to(self.device)
        # tags = [sent len, batch size]
            true_tags = batch.tag.to(self.device)
            self.optimizer.zero_grad()
            pred_tags_list, batch_loss = self.model(words, chars, true_tags)
            pred_tags_epoch += pred_tags_list
        # to calculate the loss and accuracy, we flatten both prediction and true tags
        # flatten pred_tags to [sent len, batch size, output dim]
            true_tags_epoch += [
                [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                for sent_tag in true_tags.permute(1, 0).tolist()
            ]
            # batch_acc = self.accuracy(pred_tags_list, true_tags_list)
            # batch_loss = self.loss_fn(pred_tags, true_tags)
            # batch_acc = self.accuracy(pred_tags, true_tags)
            # self.f_score(pred_tags, true_tags)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch)
        return epoch_loss / len(self.data.train_iter), epoch_score

    def evaluate(self, iterator):
        epoch_loss = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        idx2tag = self.data.tag_field.vocab.itos
        self.model.eval()
        with torch.no_grad():
          # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)
                chars = batch.char.to(self.device)
                true_tags = batch.tag.to(self.device)
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                pred_tags_epoch += pred_tags
                true_tags_epoch += [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]

                epoch_loss += batch_loss.item()
                # print(pred_tags, true_tags_list)
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch)
        return epoch_loss / len(iterator), epoch_score

    def train_live(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_f1 = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn f1: {train_f1 * 100:.2f}%")
            val_loss, val_f1 = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val f1: {val_f1 * 100:.2f}%")
        test_loss, test_f1 = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test f1: {test_f1 * 100:.2f}%")
    
    def train(self, n_epochs):
        history = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
        }
        elapsed_train_time = 0
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_f1 = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            val_loss, val_f1 = self.evaluate(self.data.val_iter)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
        test_loss, test_f1 = self.evaluate(self.data.test_iter)
        history["test_loss"] = test_loss
        history["test_f1"] = test_f1
        history["elapsed_train_time"] = elapsed_train_time
        return history

    # @staticmethod
    # def visualize_attn(tokens, weights):
