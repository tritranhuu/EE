
def get_configs(corpus, device):
    base = {
        "word_input_dim": len(corpus.word_field.vocab),
        "char_pad_idx": corpus.char_pad_idx,
        "word_pad_idx": corpus.word_pad_idx,
        "tag_names": corpus.tag_field.vocab.itos,
        "device": device,
        'data': corpus
    }
    w2v = {
        "word_emb_pretrained": corpus.word_field.vocab.vectors if corpus.wv_model else None
    }
    char_cnn = {
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
    cnn_seq = {
        "model_arch": "cnn_seq",
        "cnn_out_channel": 100,
        "cnn_kernels": [3,5,7],
        "cnn_dropout": 0.25
    }

    cnn_trig = {
        "model_arch": "cnn_trig",
        "cnn_out_channel": 100,
        "cnn_kernels": [3,4,5],
        "cnn_dropout": 0.25,
        # "pos_emb_size": 200,
        # "pos_emb_dim": 25
    }
    pos_emb = {
        "pos_emb_size": 200,
        "pos_emb_dim": 25
    }

    crf = {
      "use_crf" : True
    }

    # this is the main config, which is based on the previous building blocks
    configs = {
        "bilstm": base,
        "bilstm+w2v": {**base, **w2v},
        "bilstm+w2v+charcnn": {**base, **w2v, **char_cnn},
        "bilstm+w2v+charcnn+attn": {**base, **w2v, **char_cnn, **attn},
  
        "cnn_seq": {**base, **cnn_seq},
        "cnn_seq+w2v": {**base, **cnn_seq, **w2v},
        "cnn_seq+w2v+charcnn": {**base, **cnn_seq, **w2v, **char_cnn},

        "cnn_trig": {**base, **cnn_trig},
        # "cnn_trig+w2v": {**base, **cnn_trig, **w2v},
        # "cnn_trig+w2v+charcnn": {**base, **cnn_trig, **w2v, **char_cnn},

        "cnn_trig+w2v+position": {**base, **cnn_trig, **w2v, **pos_emb},
        # "transformer+w2v+cnn": {**base, **transformer, **w2v, **char_cnn, **attn}
    }
    return configs

def get_configs_arguments(corpus, device):
    base = {
        "word_input_dim": len(corpus.word_field.vocab),
        "char_pad_idx": corpus.char_pad_idx,
        "word_pad_idx": corpus.word_pad_idx,
        "entity_pad_idx": corpus.entity_pad_idx,
        "event_pad_idx": corpus.event_pad_idx,
        "argument_names": corpus.argument_field.vocab.itos,
        "device": device
    }
    char_cnn = {
        "use_char_emb": True,
        "char_input_dim": len(corpus.char_field.vocab),
        "char_emb_dim": 25,
        "char_emb_dropout": 0.25,
        "char_cnn_filter_num": 4,
        "char_cnn_kernel_size": 3,
        "char_cnn_dropout": 0.25
    }
    w2v = {
        "word_emb_pretrained": corpus.word_field.vocab.vectors if corpus.wv_model else None
    }
    entity_emb = {
        "entity_emb_size":len(corpus.entity_field.vocab),
        "entity_emb_dim":25,
        "entity_emb_dropout":0.25
    }
    event_emb = {
        "event_emb_size":len(corpus.event_field.vocab),
        "event_emb_dim":25,
        "event_emb_dropout":0.25
    }
    cnn_trig = {
        "model_arch": "cnn_trig",
        "cnn_out_channel": 100,
        "cnn_kernels": [3,4,5],
        "cnn_dropout": 0.25,
        # "pos_emb_size": 200,
        # "pos_emb_dim": 25
    }
    pos_emb = {
        "pos_emb_size": 200,
        "pos_emb_dim": 25
    }

    configs = {
        "bilstm": {**base, **w2v, **char_cnn, **entity_emb, **event_emb},
        "cnn": {**base, **cnn_trig, **w2v, **char_cnn, **entity_emb, **event_emb}
    }
    return configs