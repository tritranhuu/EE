import tensorflow as tf
import numpy as np

class LSTMModel():
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])

        fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(args.num_layers)], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(args.num_layers)], state_is_tuple=True)
        
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)

        check = tf.transpose(self.input_data, perm=[1,0,2])

        (output,_) = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell,
            self.input_data,sequence_length=self.length, dtype=tf.float32
        )
        
        out_fw, out_bw = output
        output = tf.concat([out_fw, out_bw], axis=-1)

        weight, bias = self.weight_and_bias(2*args.rnn_size, args.class_size)

        output = tf.reshape(output, [-1, 2*args.rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)

        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])

        self.loss = self
    
    def lstm_cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.args.rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)
        

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

