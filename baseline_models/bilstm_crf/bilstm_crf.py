import tensorflow as tf
import tensorflow_addons as tf_ad

class BiLstmModel(tf.keras.Model):
    def __init__(self, args, vocab_size):
        super(BiLstmModel, self).__init__()

        self.num_hidden = args.num_hidden
        # self.vocab_size = args.vocab_size
        self.class_size = args.class_size
        self.args = args

        self.embedding = tf.keras.layers.Embedding(vocab_size, args.embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.num_hidden, return_sequences=True))
        self.dense  = tf.keras.layers.Dense(args.class_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(args.class_size, args.class_size)))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.optimizer = tf.keras.optimizers.Adam(args.lr)

    def call(self, input_data, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(input_data, 0), dtype=tf.int32), axis=-1)
        # words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data), axis=2))
        # text_lens = tf.cast(tf.reduce_sum(words_used_in_sent, axis=1), tf.int32)
        inputs = self.embedding(input_data)
        inputs = tf.keras.layers.Input(shape=(self.args.sentence_length, self.args.word_dim))
        inputs = self.dropout(input_data, training)
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequence = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequence, text_lens, transition_params=self.transition_params)

            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens

    def train_one_step(self, inp_batch, out_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = self.call(inp_batch, out_batch, training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, logits, text_lens