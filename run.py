from data_utils.corpus import Corpus
from deeplearning_models.bilstm_crf import BiLSTM_CRF
from trainer.event_detection_for_crf import ED

from torch.optim import Adam
from torch import nn

if __name__ == "__main__":
    corpus = Corpus(
        input_folder="./data/csv",
        min_word_freq=3,  
        batch_size=64
    )
    model = BiLSTM_CRF(
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
    ed.train(10)