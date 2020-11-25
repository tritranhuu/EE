from data_utils.corpus import Corpus

from models_deeplearning.argument_detection import Model_EA
from models_deeplearning.event_detections import Model_ED

from trainer.event_extraction_trainer import Trainer

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import nn

import argparse
from configs import get_configs, get_configs_arguments


if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="./data/data_ace/arg_data")
    parser.add_argument('--min_word_freq', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--wv_file', type=str, default='./pretrained_embedding/word2vec/wiki-news-300d-1M.vec')
    
    corpus = Corpus(
        parser.parse_args()
    )

    configs_event = get_configs(corpus, device)
    configs_arg = get_configs_arguments(corpus, device)


    model_event = Model_ED(**configs_event['cnn_seq+w2v'])
    model_arg = Model_EA(**configs['cnn+w2v'])

    trainer = Trainer(
        model_event=model_event,
        model_arg=model_arg,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss,
        device = device
        )

    trainer.test()