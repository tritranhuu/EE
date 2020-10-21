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
from torchcrf import CRF

import numpy as np
from sklearn.metrics import classification_report

class Corpus(object):

  def __init__(self, input_folder, min_word_freq, batch_size, wv_file=None):
    # list all the fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)
    self.char_nesting_field = Field(tokenize=list)
    self.char_field = NestedField(self.char_nesting_field)

    # create dataset using built-in parser from torchtext
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=input_folder,
        train="train.csv",
        validation="dev.csv",
        test="test.csv",
        fields=(
          (("word", "char"), (self.word_field, self.char_field)), 
          ("tag", self.tag_field))
    )
    # convert fields to vocabulary list
    # self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)
    # create iterator for batch input
    
    
    if wv_file:
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_file, binary=True)
        self.embedding_dim = self.wv_model.vector_size
        word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
        word_counter = Counter(word_freq)
        self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            # mapping each vector/embedding from word2vec model to word_field vocabs
        vectors = []
        for word, idx in self.word_field.vocab.stoi.items():
            if word in self.wv_model.wv.vocab.keys():
                vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
            else:
                vectors.append(torch.zeros(self.embedding_dim))
        self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                # list of vector embedding, orderred according to word_field.vocab
                vectors=vectors,
                dim=self.embedding_dim
            )
    else:
        self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.char_field.build_vocab(self.train_dataset.char)

    self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
        batch_size=batch_size
    ) 
    
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
    self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]

class BiLSTM(nn.Module):

  def __init__(self, 
                input_dim, 
                embedding_dim, 
                hidden_dim, 
                char_emb_dim,
                char_input_dim,
                char_cnn_filter_num,
                char_cnn_kernel_size,
                output_dim, 
                lstm_layers,
                emb_dropout,
                cnn_dropout, 
                lstm_dropout, 
                fc_dropout, 
                word_pad_idx,
                char_pad_idx,
                tag_pad_idx):
    super().__init__()
    self.embedding_dim = embedding_dim
    # LAYER 1: Embedding
    self.embedding = nn.Embedding(
        num_embeddings=input_dim, 
        embedding_dim=embedding_dim, 
        padding_idx=word_pad_idx
    )
    self.emb_dropout = nn.Dropout(emb_dropout)
    # LAYER 1B: Char Embedding-CNN
    self.char_emb_dim = char_emb_dim
    self.char_emb = nn.Embedding(
        num_embeddings=char_input_dim,
        embedding_dim=char_emb_dim,
        padding_idx=char_pad_idx
    )
    self.char_cnn = nn.Conv1d(
        in_channels=char_emb_dim,
        out_channels=char_emb_dim * char_cnn_filter_num,
        kernel_size=char_cnn_kernel_size,
        groups=char_emb_dim  # different 1d conv for each embedding dim
    )
    self.cnn_dropout = nn.Dropout(cnn_dropout)
    # LAYER 2: BiLSTM
    self.lstm = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=hidden_dim,
        num_layers=lstm_layers,
        bidirectional=True,
        dropout=lstm_dropout if lstm_layers > 1 else 0
    )
    # LAYER 3: Fully-connected
    self.fc_dropout = nn.Dropout(fc_dropout)
    # self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional
    self.fc = self.get_linear_layer(hidden_dim * 2, output_dim)

    # LAYER 4: CRF
    self.tag_pad_idx = tag_pad_idx
    self.crf = CRF(num_tags=output_dim)
    # for name, param in self.named_parameters


  def forward(self, words, chars, tags=None):
    # words = [sentence length, batch size]
    # chars = [batch size, sentence length, word length)
    # embedding_out = [sentence length, batch size, embedding dim]
    embedding_out = self.emb_dropout(self.embedding(words))
    # lstm_out = [sentence length, batch size, hidden dim * 2]
    # char_emb_out = self.emb_dropout(self.char_emb(chars))
    # batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
    # char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
    #     # for character embedding, we need to iterate over sentences
    # for sent_i in range(sent_len):
    #   sent_char_emb = char_emb_out[:, sent_i,:,:]
    #   sent_char_emb_p = sent_char_emb.permute(0,2,1)
    #   char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
    #   char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
    # char_cnn = self.cnn_dropout(char_cnn_max_out)
    # char_cnn_p = char_cnn.permute(1,0,2)
    # word_features = torch.cat((embedding_out, char_cnn_p), dim=2)
    
    lstm_out, _ = self.lstm(embedding_out)
    # ner_out = [sentence length, batch size, output dim]
    fc_out = self.fc(self.fc_dropout(lstm_out))

    if tags is not None:
        mask = tags != self.tag_pad_idx
        crf_out = self.crf.decode(fc_out, mask=mask)
        crf_loss = -self.crf(fc_out, tags=tags, mask=mask, reduction='sum')
    else:
        crf_out = self.crf.decode(fc_out)
        crf_loss = None

    return crf_out, crf_loss

  def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
    for name, param in self.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

  def init_embeddings(self, char_pad_idx, word_pad_idx, pretrained=None, freeze=True):
    # initialize embedding for padding as zero
    self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
    self.char_emb.weight.data[char_pad_idx] = torch.zeros(self.char_emb_dim)
    if pretrained is not None:
        self.embedding = nn.Embedding.from_pretrained(
          embeddings=torch.as_tensor(pretrained),
          padding_idx=word_pad_idx,
          freeze=freeze
        )
  @staticmethod
  def get_linear_layer(input_dim, output_dim):
    linear_layer = nn.Linear(input_dim, output_dim, bias=True)
    mean = 0.0  # std_dev = np.sqrt(variance)
    std_dev = np.sqrt(2 / (output_dim + input_dim))  # np.sqrt(1 / m) # np.sqrt(1 / n)
    weight = np.random.normal(mean, std_dev, size=(output_dim, input_dim)).astype(np.float32)
    std_dev = np.sqrt(1 / output_dim)  # np.sqrt(2 / (m + 1))
    bt = np.random.normal(mean, std_dev, size=output_dim).astype(np.float32)

    linear_layer.weight.data = torch.tensor(weight, requires_grad=True)
    linear_layer.bias.data = torch.tensor(bt, requires_grad=True)
    return linear_layer

#   def init_crf_transitions(self, tag_names, imp_value=-100):
#       num_tags = len(tag_names)
#       for i in range(num_tags):



  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NER(object):

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
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        # print(flatten_y)
        correct = [pred==tag for pred, tag in zip(flatten_preds, flatten_y)]
        return sum(correct) / len(correct) if len(correct) > 0 else 0
    
    def f_score(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()
        y_pred = []
        y_true = []
        for i in range(max_preds[non_pad_elements].shape[0]):
            y_pred.append(self.data.tag_field.vocab.itos[max_preds[non_pad_elements][i].item()])
            y_true.append(self.data.tag_field.vocab.itos[y[non_pad_elements][i].item()])
        print(classification_report(y_pred=np.array(y_pred), y_true=np.array(y_true), labels=list(self.data.tag_field.vocab.stoi.keys())[2:]))

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
            pred_tags_list, batch_loss = self.model(words, chars, true_tags)
        # to calculate the loss and accuracy, we flatten both prediction and true tags
        # flatten pred_tags to [sent len, batch size, output dim]
            true_tags_list = [
                [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                for sent_tag in true_tags.permute(1, 0).tolist()
            ]
            batch_acc = self.accuracy(pred_tags_list, true_tags_list)
            # batch_loss = self.loss_fn(pred_tags, true_tags)
            # batch_acc = self.accuracy(pred_tags, true_tags)
            # self.f_score(pred_tags, true_tags)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc
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
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                true_tags_list = [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                batch_acc = self.accuracy(pred_tags, true_tags_list)
                epoch_acc += batch_acc

                epoch_loss += batch_loss.item()
                # print(pred_tags, true_tags_list)
                flatten_preds = [pred for sent_pred in pred_tags for pred in sent_pred]
                flatten_y = [tag for sent_tag in true_tags_list for tag in sent_tag]
                y_pred.extend([self.data.tag_field.vocab.itos[x] for x in flatten_preds])
                y_true.extend([self.data.tag_field.vocab.itos[x] for x in flatten_y])
                
        print(classification_report(y_pred=y_pred, y_true=y_true, labels=list(self.data.tag_field.vocab.stoi.keys())[2:]))
    
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

  # main training sequence
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = NER.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    corpus = Corpus(
        input_folder="./data/",
        min_word_freq=2,  # any words occurring less than 3 times will be ignored from vocab
        batch_size=128,
        wv_file = './pretrained_embedding/word2vec/baomoi.vn.model.bin'
    )
    bilstm = BiLSTM(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=300,
        char_emb_dim=25,
        char_input_dim=len(corpus.char_field.vocab),
        char_cnn_filter_num=5,
        char_cnn_kernel_size=3,
        hidden_dim=64,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=2,
        emb_dropout=0.5,
        cnn_dropout=0.25,
        lstm_dropout=0.1,
        fc_dropout=0.25,
        word_pad_idx=corpus.word_pad_idx,
        char_pad_idx=corpus.char_pad_idx,
        tag_pad_idx=corpus.tag_pad_idx
    )
    bilstm.init_weights()
    bilstm.init_embeddings(
      char_pad_idx=corpus.char_pad_idx,
      word_pad_idx=corpus.word_pad_idx,
      pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
      freeze=True
      )
    ner = NER(
        model=bilstm,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
        )
    ner.train(20)