from data_utils.corpus import Corpus

from models_deeplearning.argument_detection import Model_EA
from models_deeplearning.event_detections import Model_ED

from trainer.event_extraction_trainer import Trainer

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import nn

import argparse
import pickle as pkl
from configs import get_configs, get_configs_arguments


if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="./data/arg_data")
    parser.add_argument('--min_word_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--wv_file', type=str, default='./pretrained_embedding/word2vec/baomoi.vn.model.bin')
    
    corpus = Corpus(
        parser.parse_args()
    )

    # filehandler = open("./pretrained_model/corpus.obj","wb")
    # pkl.dump(corpus,filehandler)

    configs_event = get_configs(corpus, device)
    configs_arg = get_configs_arguments(corpus, device)


    model_event = Model_ED(**configs_event['cnn_seq+w2v'])
    model_arg = Model_EA(**configs_arg['cnn+w2v'])

    model_event.load_state('./pretrained_model/ace/cnn_seq+w2v_ED.pt')
    model_arg.load_state('./pretrained_model/cnn+w2v_EA.pt')

    trainer = Trainer(
        model_event=model_event,
        model_arg=model_arg,
        data=corpus,
        optimizer_cls=Adam,
        loss_fn_cls=nn.CrossEntropyLoss,
        device = device
        )

    trainer.test()