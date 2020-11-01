from data_utils.corpus import Corpus
from models_deeplearning.event_detections import Model
from trainer.trainer_with_crf import Trainer

import torch
from torch.optim import Adam
from torch import nn

import argparse
from configs import get_configs


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
    configs = get_configs(corpus, device)
    model_name = "bilstm+w2v+cnn"
    model = Model(**configs[model_name])

    trainer = Trainer(
        model=model,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss,
        device = device
        )

    trainer.train_live(20)

