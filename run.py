from data_utils.corpus import Corpus
from deeplearning_models.cnn_trigger_candidates import CNN
from trainer.event_detection import ED

import torch
from torch.optim import Adam
from torch import nn

import argparse



if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="./data/csv")
    parser.add_argument('--min_word_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--wv_file', type=str, default='./pretrained_embedding/word2vec/baomoi.vn.model.bin')
    
    corpus = Corpus(
        parser.parse_args()
    )

    parser.add_argument('--input_dim', type=int, default=len(corpus.word_field.vocab))
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--char_emb_dim', type=int, default=25)
    parser.add_argument('--char_input_dim', type=int, default=len(corpus.char_field.vocab))
    parser.add_argument('--char_cnn_filter_num', type=int, default=5, help="path of saved model")
    parser.add_argument('--char_cnn_kernel_size', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=corpus.tag_field.vocab)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--attn_heads', type=int, default=16)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--cnn_dropout', type=float, default=0.25)
    parser.add_argument('--char_cnn_dropout', type=float, default=0.25)
    parser.add_argument('--lstm_dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.25)
    parser.add_argument('--fc_dropout', type=float, default=0.25)
    parser.add_argument('--word_pad_idx', type=int, default=corpus.word_pad_idx)
    parser.add_argument('--char_pad_idx', type=int, default=corpus.char_pad_idx)
    parser.add_argument('--tag_pad_idx', type=int, default=corpus.tag_pad_idx)
    parser.add_argument('--cnn_kernels', type=list, default=[3,4,5])
    parser.add_argument('--cnn_in_chanel', type=int, default=1)
    parser.add_argument('--cnn_out_chanel', type=int, default=100)
    
    model = BiLSTM_CRF(
        parser.parse_args()
    )
    model.init_weights()
    model.init_embeddings(
      char_pad_idx=corpus.char_pad_idx,
      word_pad_idx=corpus.word_pad_idx,
      pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
      freeze=True
      )

    ed = ED(
        model=model,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss
        )
    ed.train(20)