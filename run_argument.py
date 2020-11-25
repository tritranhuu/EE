from data_utils.corpus_arguments import CorpusArgument
from models_deeplearning.argument_detection import Model_EA
from trainer.argument_detection_trainer import Trainer
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
    parser.add_argument('--input_folder', type=str, default="./data/arg_data")
    parser.add_argument('--min_word_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--wv_file', type=str, default='./pretrained_embedding/word2vec/baomoi.vn.model.bin')
    
    corpus = CorpusArgument(
        parser.parse_args()
    )
    configs = get_configs_arguments(corpus, device)
    # model_name = "cnn+w2v+entity"
    # model = Model_EA(**configs[model_name])

    # trainer = Trainer(
    #     model=model,
    #     data=corpus,
    #     optimizer_cls=Adam,
    #     loss_fn_cls=nn.CrossEntropyLoss,
    #     device = device
    #     )

    # trainer.train_live(20)
    model_names = ['cnn+w2v']
    num_epochs = 20
    histories = {}
    for model_name in model_names:
        print(f"Start Training: {model_name}")
        trainer = Trainer(
            model=Model_EA(**configs[model_name]),
            data=corpus,
            optimizer_cls=Adam,
            loss_fn_cls=nn.CrossEntropyLoss,
            device=device,
            checkpoint_path=f"pretrained_model/{model_name}_EA.pt"
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
    plt.savefig('/content/drive/My Drive/EE/ace_arg.png')

    model_test_f1 = [(m, histories[m]["test_f1"]) for m in histories]
    model_test_f1_sorted = sorted(model_test_f1, key=lambda m: m[1])
    model_names = [m[0] for m in model_test_f1_sorted]
    y_pos = list(range(len(model_names)))
    f1_scores = [m[1] for m in model_test_f1_sorted]
    fig, ax = plt.subplots()
    _ = ax.barh(y_pos, f1_scores, align='center')
    _ = ax.set_yticks(y_pos)
    _ = ax.set_yticklabels(model_names)
    _ = ax.set_title("Test F1")
    plt.savefig('/content/drive/My Drive/EE/ace_test_arg.png')