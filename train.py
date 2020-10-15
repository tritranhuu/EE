from baseline_models.lstm.lstm_model import LSTMModel

import pickle as pkl
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse

def get_train_data():
    emb = pkl.load(open('data/train_embed.pkl', 'rb'))
    tag = pkl.load(open('data/train_tag.pkl', 'rb'))
    return emb, tag

def get_dev_data():
    emb = pkl.load(open('data/dev_embed.pkl', 'rb'))
    tag = pkl.load(open('data/dev_tag.pkl', 'rb'))
    return emb, tag


def f1(args, prediction, target, length):
    tp = np.array([0] * (args.class_size + 1))
    fp = np.array([0] * (args.class_size + 1))
    fn = np.array([0] * (args.class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = args.class_size - 1
    for i in range(args.class_size):
        if i != unnamed_entity:
            tp[args.class_size] += tp[i]
            fp[args.class_size] += fp[i]
            fn[args.class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(args.class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[args.class_size]

def train(args):
    train_inp, train_out = get_train_data()
    dev_inp, dev_out = get_dev_data()
    print(np.array(train_inp[0]).shape)
    print(np.array(train_out[0]).shape)

    model = LSTMModel(args)
    print("alo")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, 'model.ckpt')
            print("model restored")

        for e in range(args.epoch):
            for ptr in range(0, len(train_inp), args.batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                          model.output_data: train_out[ptr:ptr + args.batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, "output/lstm/model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: dev_inp,
                                                                       model.output_data: dev_out})
            print("epoch %d:" % e)
            print('test_a score:')
            m = f1(args, pred, dev_inp, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, "output/model_max.ckpt")
                print("max model saved in file: %s" % save_path)
                # pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                #                                                            model.output_data: test_b_out})
                # print("test_b score:")
                # f1(args, pred, test_b_out, length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dim', type=int, default=300, help='dimension of word vector')
    parser.add_argument('--sentence_length', type=int, default=40, help='max sentence length')
    parser.add_argument('--class_size', type=int, default=34, help='number of classes')
    parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--restore', type=str, default=None, help="path of saved model")
    train(parser.parse_args())