import argparse

from embedding.word2vec import WordVec
from embedding.get_icon_embeddings import *

from prepare_data.data_utils import DataProcessor

def get_data_embedded(args):
    data = DataProcessor(args.data_dir)
    model = WordVec(args)
    get_input(model, 300, data.sentences, './data/train_embed.pkl', './data/train_tag.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='' , help='corpus location')
    parser.add_argument('--dimension', type=int, default=300, help='vector dimension')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--vocab_size', type=int, default= 5000, help='vocabulary size')
    parser.add_argument('--workers', type=int, default=3, help='number of threads')
    parser.add_argument('--sg', type=int, default=1, help='if skipgram 1 if cbow 0')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size of training')
    parser.add_argument('--sentence_length', type=int, default= 30, help='sentence length')
    parser.add_argument('--restore', type=str, default='pretrained_embedding/word2vec/baomoi.vn.model.bin', help='word2vec format save')
    parser.add_argument('--data_dir', type=str, default='data/train', help='data dir')
    args = parser.parse_args()

    get_data_embedded(args)