import pandas as pd
import numpy as np
import tensorflow as tf
import keras


data_train = pd.read_csv('./data/train.txt', sep="\t")
data_dev = pd.read_csv('./data/dev.txt', sep="\t", quoting=3)
data_test = pd.read_csv('./data/test.txt', sep="\t")
print("loaded")

# words = list(set(data_train['Word'].values))
# tags = list(set(data_train['Tag'].values))

# print(len(tags))

class SentenceGetter(object):
    def __init__(self, data_train, data_dev, data_test):
        self.n_sent = 1
        self.data_train = data_train
        self.data_dev = data_dev
        self.data_test = data_test
        
        self.words = list(set(data_train['Word'].values).union(set(data_dev['Word'].values)).union(set(data_test['Word'].values)))
        self.tags = list(set(data_train['Tag'].values).union(set(data_dev['Tag'].values)).union(set(data_test['Tag'].values)))
        
        self.empty = False
        agg_func = lambda s: [(w, e, t) for w, e, t in zip(s['Word'].values.tolist(),
                                                           s['Entity'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        
        self.grouped_train = self.data_train.groupby("Sent#").apply(agg_func)
        self.sentences_train = [s for s in self.grouped_train]
        self.grouped_dev = self.data_dev.groupby("Sent#").apply(agg_func)
        self.sentences_dev = [s for s in self.grouped_dev]
        self.grouped_test = self.data_test.groupby("Sent#").apply(agg_func)
        self.sentences_test = [s for s in self.grouped_test]
        
        self.preprocess_data()
    
    def preprocess_data(self):
        self.word2idx = {w: i+2 for i, w in enumerate(self.words)}
        self.word2idx['UNK'] = 1
        self.word2idx['PAD'] = 0
        self.idx2word = {i:w for w, i in self.word2idx.items()}
        
        self.tag2idx = {t:i+1 for i, t in enumerate(self.tags)}
        self.tag2idx['PAD'] = 0
        self.idx2tag = {i:t for t, i in self.tag2idx.items()}

        X_train = [[self.word2idx[w[0]] for w in s] for s in self.sentences_train]
        self.X_train = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=X_train, padding='post', value=self.word2idx['PAD'])
        y_train = [[self.tag2idx[w[2]] for w in s] for s in self.sentences_train]
        self.y_train = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=y_train, padding='post', value=self.tag2idx['PAD'])
        self.y_train = [tf.keras.utils.to_categorical(i, num_classes=len(self.tags)+1) for i in self.y_train]
        
        X_dev = [[self.word2idx[w[0]] for w in s] for s in self.sentences_dev]
        self.X_dev = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=X_dev, padding='post', value=self.word2idx['PAD'])
        y_dev = [[self.tag2idx[w[2]] for w in s] for s in self.sentences_dev]
        self.y_dev = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=y_dev, padding='post', value=self.tag2idx['PAD'])
        self.y_dev = [tf.keras.utils.to_categorical(i, num_classes=len(self.tags)+1) for i in self.y_dev]
        
        X_test = [[self.word2idx[w[0]] for w in s] for s in self.sentences_test]
        self.X_test = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=X_test, padding='post', value=self.word2idx['PAD'])
        y_test = [[self.tag2idx[w[2]] for w in s] for s in self.sentences_test]
        self.y_test = tf.keras.preprocessing.sequence.pad_sequences(maxlen=30, sequences=y_test, padding='post', value=self.tag2idx['PAD'])
        self.y_test = [tf.keras.utils.to_categorical(i, num_classes=len(self.tags)+1) for i in self.y_test]
    
    # def get_next(self):
    #     try:
    #         s = self.grouped[self.n_sent]
    #         self.n_sent += 1
    #         return s
    #     except:
    #         return None

getter = SentenceGetter(data_train, data_dev, data_test)

input_ = tf.keras.Input(shape=(30,))
model = keras.layers.embeddings.Embedding(input_dim=len(getter.words)+2, output_dim=50, input_length=30, mask_zero=True)(input)

lstm_layer = keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.1)
model = keras.layers.Bidirectional(lstm_layer(model))

dense_layer_1 = keras.layers.Dense(50, activation='relu')
model = keras.layers.TimeDistributed(dense_layer_1(model))

dense_layer_2 = keras.layers.Dense(len(getter.tags) +1, activation='relu')
out = keras.layers.TimeDistributed(dense_layer_2(model))


model = keras.models.Model(input_, out)
model.compile(optimizer="rmsprop",loss=keras.losses.CategoricalCrossentropy)
model.fit(getter.X_train, np.array(getter.y_train),batch_size=32, epochs=5, validation_data=(getter.X_dev, np.array(getter.y_dev)), verbose=2)