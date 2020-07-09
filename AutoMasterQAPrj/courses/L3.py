
import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import argparse

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 定义损失函数
def loss_function(real, pred,pad_index):
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype) #转换为和loss_类型相同的张量
    loss_ *= mask
    return tf.reduce_mean(loss_)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.use_bi_gru = True
        # 双向
        if self.use_bi_gru:
            self.enc_units = self.enc_units // 2

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

        self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, enc_input):
        # (batch_size, enc_len, embedding_dim)
        enc_input_embedded = self.embedding(enc_input)

        initial_state = self.gru.get_initial_state(enc_input_embedded)

        if self.use_bi_gru:
            # 是否使用双向GRU
            output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=initial_state * 2)
            enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)

        else:
            # 单向GRU
            output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

        return output, enc_hidden


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input, prev_dec_hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_input = self.embedding(dec_input)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # dec_input (batch_size, 1, embedding_dim + hidden_size)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        # passing the concatenated vector to the GRU
        dec_output, dec_hidden = self.gru(dec_input)

        # dec_output shape == (batch_size * 1, hidden_size)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)
        return pred, dec_hidden


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        #  model.keras.layers.Dense

    def call(self, dec_hidden, enc_output):
        # dec_hidden shape == (batch_size, hidden size)
        # enc_output (batch_size, enc_len, enc_units)

        # hidden_with_time_axis shape == (batch_size, 1, dec_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, enc_len, attn_units)
        # 计算注意力权重值
        # score shape == (batch_size, enc_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention_weights (batch_size, enc_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # enc_output (batch_size, enc_len, enc_units)
        # attention_weights (batch_size, enc_len, 1)
        context_vector = attention_weights * enc_output

        # context_vector shape after sum == (batch_size, enc_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights



def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result

def load_word2vec(params):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return:
    """
    word2vec_dict = load_pkl(params['word2vec_output'])
    vocab_dict = open(params['vocab_path'], encoding='utf-8').readlines()
    embedding_matrix = np.zeros((params['vocab_size'], params['embed_size']))

    for line in vocab_dict[:params['vocab_size']]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        print(params["batch_size"])
        self.encoder = Encoder(vocab_size = params["vocab_size"],
                               embedding_dim = params["embed_size"],
                               embedding_matrix = self.embedding_matrix,
                               enc_units = params["enc_units"],
                               batch_size = params["batch_size"])

        self.attention = BahdanauAttention(units = params["attn_units"])

        self.decoder = Decoder(vocab_size =  params["vocab_size"],
                               embedding_dim = params["embed_size"],
                               embedding_matrix = self.embedding_matrix,
                               dec_units = params["dec_units"],
                               batch_size = params["batch_size"])

    # def call_decoder_onestep(self, dec_input, dec_hidden, enc_output):
    #     # context_vector ()
    #     # attention_weights ()
    #     context_vector, attention_weights = self.attention(dec_hidden, enc_output)
    #
    #     # pred ()
    #     pred, dec_hidden = self.decoder(dec_input,
    #                                     None,
    #                                     None,
    #                                     context_vector)
    #     return pred, dec_hidden, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)

            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            attentions.append(attn)
            #tf.concat与tf.stack这两个函数作用类似，
            # 都是在某个维度上对矩阵(向量）进行拼接，
            # 不同点在于前者拼接后的矩阵维度不变，后者则会增加一个维度。
        return tf.stack(predictions, 1), dec_hidden