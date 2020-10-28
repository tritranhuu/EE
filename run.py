from data_utils.corpus import Corpus
from deeplearning_models.deep_cnn import DeepCNN
from trainer.event_detection import ED

import torch
from torch.optim import Adam
from torch import nn

if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if __name__ == "__main__":
    corpus = Corpus(
        input_folder="./data/ace-05-brat/csv",
        min_word_freq=2,  
        batch_size=128
        # wv_file = './pretrained_embedding/word2vec/baomoi.vn.model.bin'
    )
    # model = BiLSTM_CRF(
    #     input_dim=len(corpus.word_field.vocab),
    #     embedding_dim=300,
    #     char_emb_dim=25,
    #     char_input_dim=len(corpus.char_field.vocab),
    #     char_cnn_filter_num=5,
    #     char_cnn_kernel_size=3,
    #     hidden_dim=64,
    #     output_dim=len(corpus.tag_field.vocab),
    #     lstm_layers=2,
    #     emb_dropout=0.5,
    #     cnn_dropout=0.25,
    #     lstm_dropout=0.1,
    #     fc_dropout=0.25,
    #     word_pad_idx=corpus.word_pad_idx,
    #     char_pad_idx=corpus.char_pad_idx,
    #     tag_pad_idx=corpus.tag_pad_idx
    # )
    model = DeepCNN(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=300,
        cnn_kernels = [7,3,5],
        cnn_in_chanel = 1,
        cnn_out_chanel = 100,
        char_emb_dim=25,
        char_input_dim=len(corpus.char_field.vocab),
        char_cnn_filter_num=5,
        char_cnn_kernel_size=3,
        output_dim=len(corpus.tag_field.vocab),
        emb_dropout=0.5,
        char_cnn_dropout=0.25,
        cnn_dropout=0.1,
        fc_dropout=0.25,
        word_pad_idx=corpus.word_pad_idx,
        char_pad_idx=corpus.char_pad_idx
    )
    model.init_weights()
    model.init_embeddings(
      char_pad_idx=corpus.char_pad_idx,
      word_pad_idx=corpus.word_pad_idx,
      # pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
      pretrained= None,
      freeze=True
      )
    ed = ED(
        model=model,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
        )
    ed.train(20)