import tensorflow as tf
import tensorflow_addons as tf_ad
import pickle as pkl
import numpy as np

from baseline_models.bilstm_crf.bilstm_crf import BiLstmModel


# def get_train_data():
#     emb = pkl.load(open('data/train_embed.pkl', 'rb'))
#     tag = pkl.load(open('data/train_tag.pkl', 'rb'))
#     return emb, tag

# def get_dev_data():
#     emb = pkl.load(open('data/dev_embed.pkl', 'rb'))
#     tag = pkl.load(open('data/dev_tag.pkl', 'rb'))
#     return emb, tag


def train_one_step(model, optimizer, inp_batch, out_batch):
    with tf.GradientTape() as tape:
        logits, text_lens, log_likelihood = model(inp_batch, out_batch, training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss,logits, text_lens

# def get_acc_one_step(logits, text_lens, labels_batch):
#     paths = []
#     accuracy = 0
#     for logit, text_len, labels in zip(logits, text_lens, labels_batch):
#         viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
#         paths.append(viterbi_path)
#         correct_prediction = tf.equal(
#             tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
#                                  dtype=tf.int32),
#             tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
#                                  dtype=tf.int32)
#         )
#         accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
#     accuracy = accuracy / len(paths)
#     return accuracy
