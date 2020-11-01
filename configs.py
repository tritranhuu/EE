
def get_configs(corpus, device):
    base = {
        "word_input_dim": len(corpus.word_field.vocab),
        "char_pad_idx": corpus.char_pad_idx,
        "word_pad_idx": corpus.word_pad_idx,
        "tag_names": corpus.tag_field.vocab.itos,
        "device": device
    }
    w2v = {
        "word_emb_pretrained": corpus.word_field.vocab.vectors if corpus.wv_model else None
    }
    cnn = {
        "use_char_emb": True,
        "char_input_dim": len(corpus.char_field.vocab),
        "char_emb_dim": 37,
        "char_emb_dropout": 0.25,
        "char_cnn_filter_num": 4,
        "char_cnn_kernel_size": 3,
        "char_cnn_dropout": 0.25
    }
    attn = {
        "attn_heads": 16,
        "attn_dropout": 0.25
    }
    transformer = {
        "model_arch": "transformer",
        "trf_layers": 1,
        "fc_hidden": 256,
    }
    # this is the main config, which is based on the previous building blocks
    configs = {
        "bilstm": base,
        "bilstm+w2v": {**base, **w2v},
        "bilstm+w2v+cnn": {**base, **w2v, **cnn},
        "bilstm+w2v+cnn+attn": {**base, **w2v, **cnn, **attn},
        "transformer+w2v+cnn": {**base, **transformer, **w2v, **cnn, **attn}
    }
    return configs