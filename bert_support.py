from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

import torch

class Config:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)
class RobertaBPETokenizer:
    def __init__(self, bpe_path, vocab_path):
        self._bpe = fastBPE(Config(bpe_codes=bpe_path))
        self._vocab = self._get_vocab(vocab_path)

    @property
    def cls_token(self):
        return '<s>'

    @property
    def sep_token(self):
        return self._vocab.eos_word

    @property
    def pad_token(self):
        return self._vocab.pad_word

    @property
    def pad_token_id(self):
        return self._vocab.pad_index

    @property
    def cls_token_id(self):
        return self._vocab.bos_index

    @property
    def sep_token_id(self):
        return self._vocab.eos_index

    @property
    def max_len(self):
        return 512

    def tokenize(self, text: str):
        return self._bpe.encode(text).split()

    @staticmethod
    def _get_vocab(vocab_path):
        d = Dictionary()
        d.add_from_file(vocab_path)
        return d

    def convert_tokens_to_ids(self, tokens: list):
        return self._vocab.encode_line(
            tokens,
            line_tokenizer=lambda x: x,
            append_eos=False,
            add_if_not_exist=False,
        ).long()

    def decode(self, x: torch.Tensor):
        return self._bpe.decode(self._vocab.string(x))

