from data_utils.corpus import Corpus
from models_deeplearning.event_detections import Model
from trainer.trainer import Trainer
import matplotlib.pyplot as plt

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
    # model_name = "cnn_seq+w2v"
    # model = Model(**configs[model_name])

    # trainer = Trainer(
    #     model=model,
    #     data=corpus,
    #     optimizer_cls=Adam,
    #     loss_fn_cls=nn.CrossEntropyLoss,
    #     device = device
    #     )

    # trainer.train_live(20)

    num_epochs = 25
    histories = {}
    for model_name in configs:
        print(f"Start Training: {model_name}")
        trainer = Trainer(
            model=Model(**configs[model_name]),
            data=corpus,
            optimizer_cls=Adam,
            loss_fn_cls=nn.CrossEntropyLoss,
            device=device
        )
        histories[model_name] = trainer.train(num_epochs)
        print(f"Done Training: {model_name}")
    max_len_model_name = max([len(m) for m in histories])
    print(f"{'MODEL NAME'.ljust(max_len_model_name)}\t{'NUM PARAMS'.ljust(10)}\tTRAINING TIME")
    for model_name, history in histories.items():
        print(f"{model_name.ljust(max_len_model_name)}\t{history['num_params']:,}\t{int(history['elapsed_train_time']//60)}m {int(history['elapsed_train_time'] % 60)}s")
    epochs = [i+1 for i in range(num_epochs)]
    fig, axs = plt.subplots(2, 1, figsize=(num_epochs, 12))
    for model_name in histories:
        axs[0].plot(epochs, histories[model_name]["val_loss"], label=model_name)
        axs[1].plot(epochs, histories[model_name]["val_f1"], label=model_name)
    _ = axs[0].set_title("Val Loss")
    _ = axs[1].set_title("Val F1")
    _ = axs[1].set_xlabel("epochs")
    _ = axs[0].set_ylabel("loss")
    _ = axs[1].set_ylabel("F1")
    _ = axs[0].legend(loc="upper right")
    _ = axs[1].legend(loc="lower right")
    plt.savefig('/content/drive/My Drive/EE/fig_nocrf.png')