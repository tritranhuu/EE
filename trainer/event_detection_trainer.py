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

    def __init__(self, model, data, optimizer_cls, loss_fn_cls, device, checkpoint_path=None):
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())
        self.loss_fn = loss_fn_cls(ignore_index=self.data.event_pad_idx)
        self.checkpoint_path = checkpoint_path
    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds, y):
        preds= preds.to(self.device)
        y = y.to(self.device)
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != 0).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        # print(max_preds[non_pad_elements][1].item())
        return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

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
        epoch_acc = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        idx2tag = self.data.event_field.vocab.itos
        self.model.train()
        for batch in self.data.train_iter:
        # words = [sent len, batch size]
            words = batch.word.to(self.device)
            chars = batch.char.to(self.device)
        # tags = [sent len, batch size]
            true_tags = batch.event.to(self.device)
            self.optimizer.zero_grad()
            pred_tags, _ = self.model(words, chars)
            x = pred_tags.argmax(dim=2).permute(1, 0)
            # print(x.shape)
            # print([(t>1).nonzero().reshape(-1) for t in x])
            # print(torch.cat([(t>1).nonzero() for t in x], dim=1).shape)
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
                chars = batch.char.to(self.device)
                true_tags = batch.event.to(self.device)
                pred_tags, _ = self.model(words, chars)
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                
                pred_tags_epoch.extend([idx2tag[i] for i in pred_tags.argmax(dim=1).cpu().numpy()])
                true_tags_epoch.extend([idx2tag[i] for i in true_tags.cpu().numpy()])
        epoch_score = f1_score(true_tags_epoch, pred_tags_epoch, average="micro",labels=list(self.data.event_field.vocab.stoi.keys())[2:])
        epoch_p1 = precision_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.argument_field.vocab.stoi.keys())[2:])
        epoch_r1 = recall_score(true_tags_epoch,pred_tags_epoch, average="micro",labels=list(self.data.argument_field.vocab.stoi.keys())[2:])
        
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
