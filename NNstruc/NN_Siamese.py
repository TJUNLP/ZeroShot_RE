# coding:utf-8

from keras.layers.core import Dropout, RepeatVector, Reshape, Activation
from keras.layers.merge import concatenate, add, subtract, average, maximum
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import merge, Lambda, Flatten, Permute, multiply
from keras.layers.merge import dot
from keras.layers.merge import dot, Dot
from keras_ordered_neurons import ONLSTM
from keras.engine import Layer
from tensorflow.python.framework import ops
import tensorflow as tf
import flipGradientTF


def Model_ONBiLSTM_directMAP_tripletloss_Hloss_05_at01_allexp_2m(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])))([wrong_cos, right_cos, at_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_directMAP_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_fourloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]) + (1. - x[2]))([wrong_cos, right_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_Dy_fourloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    real_ap_cos = Dot(axes=-1, normalize=True)([tag_embedding_p, tag_embedding_a])
    real_an_cos = Dot(axes=-1, normalize=True)([tag_embedding_n, tag_embedding_a])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu((x[3] - x[4] - K.epsilon()) + x[0] - x[1]) + (1. - x[2]))([wrong_cos, right_cos, anchor_cos, real_ap_cos, real_an_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_two_fourloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]) + K.relu(margin + x[1] - x[2]))([wrong_cos, right_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_chain_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]) *
                            K.round(K.maximum(0.5 - K.epsilon(), 0.5 - (margin2 + x[2] - x[3]))) +
                            K.relu(margin1 + x[0] - x[1]) *
                            K.round(K.maximum(0.5 - K.epsilon(), 0.5 - (margin2 + x[2] - x[3]))) *
                            K.round(K.maximum(0.5 - K.epsilon(), 0.5 - (margin3 + x[1] - x[3]))))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_chain_3(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin2 + x[2] - x[3]) +
                            (K.relu(margin3 + x[1] - x[3]) +
                            K.relu(margin1 + x[0] - x[1])) *
                            K.round(K.maximum(0.5 - K.epsilon(), 0.5 - (margin2 + x[2] - x[3]))))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_chain_2(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.exp((margin2 + x[2] - x[3]) / (margin2 + 2.)) *
                            K.relu(margin2 + x[2] - x[3]) +
                            K.round(K.maximum(0.5 - K.epsilon(), 0.5 - (margin2 + x[2] - x[3]))) *
                            (K.relu(margin3 + x[1] - x[3]) +
                             K.relu(margin1 + x[0] - x[1])))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_1_ed(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    anchor_cos = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[1] - x[0]) +
                            K.relu(margin2 + x[3] - x[2]) +
                            K.relu(margin3 + x[3] - x[1]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_1_lr(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dense(1, activation='sigmoid', name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dense(1, activation='sigmoid', name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    anchor_cos = Dense(1, activation='sigmoid', name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dense(1, activation='sigmoid', name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_BiLSTM_RankMAP_three_triloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_Atten_RankMAP_three_triloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    ONBiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2, return_sequences=True), merge_mode='concat')



    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    ONBiLSTM_x1 = ONBiLSTM_layer(embedding_x1)
    ONBiLSTM_x1 = Dropout(0.25)(ONBiLSTM_x1)

    attention = Dense(1, activation='tanh')(ONBiLSTM_x1)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(200)(attention)
    attention = Permute([2, 1])(attention)
    Attention = multiply([ONBiLSTM_x1, attention])

    ONBiLSTM_x2 = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')(Attention)
    ONBiLSTM_x2 = Dropout(0.25)(ONBiLSTM_x2)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([ONBiLSTM_x2, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([ONBiLSTM_x2, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([ONBiLSTM_x2, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([ONBiLSTM_x2, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_triloss_3(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.RMSprop(lr=0.001))

    return mymodel



def Model_ONBiLSTM_RankMAP_three_triloss_2(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin1=0.1, margin2=0.1, margin3=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([BiLSTM_x1, tag_embedding_n0])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin1 + x[0] - x[1]) +
                            K.relu(margin2 + x[2] - x[3]) +
                            K.relu(margin3 + x[1] - x[3]))\
        ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_three_fourloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]) + K.relu(margin + x[1] - x[2]) + (1. - x[2]))([wrong_cos, right_cos, anchor_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_RankMAP_fourloss_Classify_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    # anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([BiLSTM_x1, tag_embedding_a])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='triloss')([wrong_cos, right_cos])

    anchor_classify = Dense(120, activation='softmax', name='anchor_classify')(BiLSTM_x1)

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a], [loss, anchor_classify])

    mymodel.compile(loss={'triloss': lambda y_true, y_pred: y_pred, 'anchor_classify': 'categorical_crossentropy'},
                    loss_weights={'triloss': 1., 'anchor_classify': 1.},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'triloss': [], 'anchor_classify': ['acc']})

    return mymodel


def Model_ONBiLSTM_RankMAP_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                          output_dim=w2v_k,
                                          input_length=input_sent_lenth,
                                          mask_zero=True,
                                          trainable=True,
                                          weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)

    word_embedding_sent_x2 = word_embedding_sent_layer(word_input_sent_x2)
    word_embedding_sent_x2 = Dropout(0.25)(word_embedding_sent_x2)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                                                          output_dim=c2v_k,
                                                          batch_input_shape=(
                                                          batch_size, input_sent_lenth, input_maxword_length),
                                                          mask_zero=False,
                                                          trainable=True,
                                                          weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)


    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=input_sent_lenth,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)

    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    input_tag = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding = tag_embedding_layer(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x2, tag_embedding])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    tloss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     input_tag], tloss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel



def Model_ONBiLSTM_RankMAP_tripletloss_1_nochar(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                          output_dim=w2v_k,
                                          input_length=input_sent_lenth,
                                          mask_zero=True,
                                          trainable=True,
                                          weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)

    word_embedding_sent_x2 = word_embedding_sent_layer(word_input_sent_x2)
    word_embedding_sent_x2 = Dropout(0.25)(word_embedding_sent_x2)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    # char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
    #                                                       output_dim=c2v_k,
    #                                                       batch_input_shape=(
    #                                                       batch_size, input_sent_lenth, input_maxword_length),
    #                                                       mask_zero=False,
    #                                                       trainable=True,
    #                                                       weights=[char_W]))
    #
    # char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    # char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    #
    # char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    #
    # char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    # char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    # char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)
    #
    # char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    # char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    # char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)


    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=input_sent_lenth,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)

    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    # embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
    #                             embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    embedding_x1 = concatenate([word_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    # embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
    #                             embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    embedding_x2 = concatenate([word_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    input_tag = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding = tag_embedding_layer(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x2, tag_embedding])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2

    tloss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     input_tag], tloss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_directMAP_AL_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='concat')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)

    mlp1_layer = Dense(100, activation='tanh')
    mlp2_layer = Dense(200, activation='tanh')
    dp_layer = Dropout(0.5)

    p_mlp1 = mlp1_layer(tag_embedding_p)
    p_mlp1 = dp_layer(p_mlp1)
    p_mlp2 = mlp2_layer(p_mlp1)
    p_mlp2 = dp_layer(p_mlp2)

    n_mlp1 = mlp1_layer(tag_embedding_n)
    n_mlp1 = dp_layer(n_mlp1)
    n_mlp2 = mlp2_layer(n_mlp1)
    n_mlp2 = dp_layer(n_mlp2)


    Flip = flipGradientTF.GradientReversal(hp_lambda=0.2)
    unseen_output_layer = Dense(2, activation='softmax', name='ISunseen_Classifier')
    dann_in = Flip(n_mlp2)
    unseen_output = unseen_output_layer(dann_in)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, p_mlp2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, n_mlp2])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([p_mlp2, n_mlp2])


    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], [loss, unseen_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'ISunseen_Classifier': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'ISunseen_Classifier': 1.},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'ISunseen_Classifier': ['acc']})
    return mymodel


def Model_ONBiLSTM_directMAPbyLSTM_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2, return_sequences=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)


    p_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x1, tag_embedding_p])
    # pinjie_p = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_p])
    n_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_n, BiLSTM_x1, tag_embedding_n])
    # pinjie_n = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_n])

    tag_BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=False), merge_mode='concat')
    tag_p = tag_BiLSTM_layer(p_pinjie)
    tag_p = Dropout(0.3)(tag_p)
    tag_n = tag_BiLSTM_layer(n_pinjie)
    tag_n = Dropout(0.3)(tag_n)
    cos_layer = Dense(1, activation='sigmoid', name='right_cos')
    right_cos = cos_layer(tag_p)
    wrong_cos = cos_layer(tag_n)


    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel



def Model_ONBiLSTM_directClassifybyLSTM_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                          output_dim=w2v_k,
                                          input_length=input_sent_lenth,
                                          mask_zero=True,
                                          trainable=True,
                                          weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)

    word_embedding_sent_x2 = word_embedding_sent_layer(word_input_sent_x2)
    word_embedding_sent_x2 = Dropout(0.25)(word_embedding_sent_x2)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                                                          output_dim=c2v_k,
                                                          batch_input_shape=(
                                                          batch_size, input_sent_lenth, input_maxword_length),
                                                          mask_zero=False,
                                                          trainable=True,
                                                          weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)


    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=input_sent_lenth,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)

    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2, return_sequences=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)


    p_x1_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x1, tag_embedding_p])
    n_x1_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_n, BiLSTM_x1, tag_embedding_n])
    # pinjie_p = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_p])
    p_x2_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x2, tag_embedding_p])
    n_x2_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_n, BiLSTM_x2, tag_embedding_n])

    tag_BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=False), merge_mode='concat')

    tag_p_x1 = tag_BiLSTM_layer(p_x1_pinjie)
    tag_p_x1 = Dropout(0.3)(tag_p_x1)
    tag_n_x1 = tag_BiLSTM_layer(n_x1_pinjie)
    tag_n_x1 = Dropout(0.3)(tag_n_x1)
    tag_p_x2 = tag_BiLSTM_layer(p_x2_pinjie)
    tag_p_x2 = Dropout(0.3)(tag_p_x2)
    tag_n_x2 = tag_BiLSTM_layer(n_x2_pinjie)
    tag_n_x2 = Dropout(0.3)(tag_n_x2)

    cos_layer = Dense(2, activation='softmax', name='right_cos')
    right_cos_1 = cos_layer(tag_p_x1)
    wrong_cos_1 = cos_layer(tag_n_x1)
    wrong_cos_2 = cos_layer(tag_p_x2)
    right_cos_2 = cos_layer(tag_n_x2)

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     input_tag_p, input_tag_n], [right_cos_1, wrong_cos_1, wrong_cos_2, right_cos_2])

    mymodel.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001),
                    loss_weights=[1., 1., 1., 1.],
                    metrics=['acc'])

    return mymodel



def Model_ONBiLSTM_directClassifybyLSTM_3(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                          output_dim=w2v_k,
                                          input_length=input_sent_lenth,
                                          mask_zero=True,
                                          trainable=True,
                                          weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)

    word_embedding_sent_x2 = word_embedding_sent_layer(word_input_sent_x2)
    word_embedding_sent_x2 = Dropout(0.25)(word_embedding_sent_x2)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                                                          output_dim=c2v_k,
                                                          batch_input_shape=(
                                                          batch_size, input_sent_lenth, input_maxword_length),
                                                          mask_zero=False,
                                                          trainable=True,
                                                          weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)


    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=input_sent_lenth,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)

    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2, return_sequences=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)


    p_x1_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x1, tag_embedding_p])
    n_x1_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_n, BiLSTM_x1, tag_embedding_n])
    # pinjie_p = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_p])
    p_x2_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x2, tag_embedding_p])

    tag_BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=False), merge_mode='concat')

    tag_p_x1 = tag_BiLSTM_layer(p_x1_pinjie)
    tag_p_x1 = Dropout(0.3)(tag_p_x1)
    tag_n_x1 = tag_BiLSTM_layer(n_x1_pinjie)
    tag_n_x1 = Dropout(0.3)(tag_n_x1)
    tag_p_x2 = tag_BiLSTM_layer(p_x2_pinjie)
    tag_p_x2 = Dropout(0.3)(tag_p_x2)

    cos_layer = Dense(2, activation='softmax', name='right_cos')
    right_cos_1 = cos_layer(tag_p_x1)
    wrong_cos_1 = cos_layer(tag_n_x1)
    wrong_cos_2 = cos_layer(tag_p_x2)

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     input_tag_p, input_tag_n], [right_cos_1, wrong_cos_1, wrong_cos_2])

    mymodel.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adam(lr=0.001),
                    loss_weights=[1., 1., 1.],
                    metrics=['acc'])

    return mymodel



def Model_ONBiLSTM_directMAPbyMLP_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)

    mlp1_layer = Dense(200, activation='tanh')
    mlp2_layer = Dense(100, activation='tanh')
    cos_layer = Dense(1, activation='sigmoid', name='right_cos')

    p_input = concatenate([BiLSTM_x1, tag_embedding_p], axis=-1)
    p_mlp1 = mlp1_layer(p_input)
    p_mlp1 = Dropout(0.5)(p_mlp1)
    p_mlp2 = mlp2_layer(p_mlp1)
    p_mlp2 = Dropout(0.5)(p_mlp2)
    right_cos = cos_layer(p_mlp2)

    n_input = concatenate([BiLSTM_x1, tag_embedding_n], axis=-1)
    n_mlp1 = mlp1_layer(n_input)
    n_mlp1 = Dropout(0.5)(n_mlp1)
    n_mlp2 = mlp2_layer(n_mlp1)
    n_mlp2 = Dropout(0.5)(n_mlp2)
    wrong_cos = cos_layer(n_mlp2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    # right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    # wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))
    # mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'ISunseen_Classifier': 'categorical_crossentropy'},
    #                 loss_weights={'TripletLoss': 1., 'ISunseen_Classifier': 0.1},
    #                 optimizer=optimizers.Adam(lr=0.001),
    #                 metrics={'TripletLoss': [], 'ISunseen_Classifier': ['acc']})

    return mymodel


def Model_ONBiLSTM_directMAPbyMLP_AL_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)

    mlp1_layer = Dense(200, activation='tanh')
    mlp2_layer = Dense(100, activation='tanh')

    p_input = concatenate([BiLSTM_x1, tag_embedding_p], axis=-1)
    p_mlp1 = mlp1_layer(p_input)
    p_mlp1 = Dropout(0.5)(p_mlp1)
    p_mlp2 = mlp2_layer(p_mlp1)
    p_mlp2 = Dropout(0.5)(p_mlp2)
    right_cos = Dense(1, activation='sigmoid', name='right_cos')(p_mlp2)

    n_input = concatenate([BiLSTM_x1, tag_embedding_n], axis=-1)
    n_mlp1 = mlp1_layer(n_input)
    n_mlp1 = Dropout(0.5)(n_mlp1)
    n_mlp2 = mlp2_layer(n_mlp1)
    n_mlp2 = Dropout(0.5)(n_mlp2)
    wrong_cos = Dense(1, activation='sigmoid', name='wrong_cos')(n_mlp2)

    Flip = flipGradientTF.GradientReversal(hp_lambda=1.0)
    dann_in = Flip(n_mlp2)
    unseen_output = Dense(2, activation='softmax', name='ISunseen_Classifier')(dann_in)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    # right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_embedding_p])
    # wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], [loss, unseen_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))
    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'ISunseen_Classifier': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'ISunseen_Classifier': 0.1},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'ISunseen_Classifier': ['acc']})

    return mymodel


def Model_ONBiLSTM_directMAP_tripletloss_Hloss_className(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, max_l=6,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2), merge_mode='concat')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(max_l,), dtype='int32')
    input_tag_n = Input(shape=(max_l,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=max_l,
                                    mask_zero=True,
                                    trainable=False,
                                    weights=[word_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)

    tag_BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=False), merge_mode='concat')
    tag_p = tag_BiLSTM_layer(tag_embedding_p)
    tag_n = tag_BiLSTM_layer(tag_embedding_n)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_n])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_p, tag_n])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])))([wrong_cos, right_cos, at_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_ONBiLSTM_directMAPbyLSTM_tripletloss_className(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, max_l=6,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    word_embedding_sent_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    word_embedding_sent_x1 = word_embedding_sent_layer(word_input_sent_x1)
    word_embedding_sent_x1 = Dropout(0.25)(word_embedding_sent_x1)


    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)


    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)


    # BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')
    BiLSTM_layer = Bidirectional(ONLSTM(100, chunk_size=5, recurrent_dropconnect=0.2, return_sequences=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    input_tag_p = Input(shape=(max_l,), dtype='int32')
    input_tag_n = Input(shape=(max_l,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=max_l,
                                    mask_zero=True,
                                    trainable=False,
                                    weights=[word_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)

    p_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_p, BiLSTM_x1])
    # pinjie_p = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_p])
    n_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding_n, BiLSTM_x1])
    # pinjie_n = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([BiLSTM_x1, tag_embedding_n])

    tag_BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=False), merge_mode='concat')
    tag_p = tag_BiLSTM_layer(p_pinjie)
    tag_p = Dropout(0.3)(tag_p)
    tag_n = tag_BiLSTM_layer(n_pinjie)
    #!!!!!!!!!!error before
    tag_n = Dropout(0.3)(tag_n)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    # right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, tag_p])
    # wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, tag_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_p, tag_n])

    right_cos = Dense(1, activation='sigmoid', name='right_cos')(tag_p)
    wrong_cos = Dense(1, activation='sigmoid', name='wrong_cos')(tag_n)

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def  Model_BiLSTM_sent_linear__KGembed(wordvocabsize, tagvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,
                    batch_size=32):

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=False,
                                    weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.25)(word_embedding_sent)

    char_input_sent = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_sent)

    char_cnn_sent = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_sent = char_cnn_sent(char_embedding_sent)
    char_embedding_sent = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent)
    char_embedding_sent = Dropout(0.25)(char_embedding_sent)

    input_e1_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e1_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e1_posi)

    input_e2_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e2_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e2_posi)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)

    embedding_x1 = concatenate([word_embedding_sent, char_embedding_sent, embedding_e1_posi, embedding_e2_posi], axis=-1)
    BiLSTM_x1 = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # attention_probs = Activation('softmax')(attention_concat)
    attention_probs = Dense(1, activation='softmax')(attention_self)
    # attention_probs = Flatten()(attention_probs)
    # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    attention_x1 = Dropout(0.25)(representation)

    mlp_x2_0 = Flatten()(tag_embedding)
    mlp_x1_1 = Dense(200, activation=None)(attention_x1)
    mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    mlp_x1_2 = Dense(100, activation=None)(mlp_x1_1)
    # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    distance = dot([mlp_x1_2, mlp_x2_0], axes=-1, normalize=True)

    mymodel = Model([word_input_sent, input_e1_posi, input_e2_posi, input_tag, char_input_sent], distance)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM_sent_MLP__KGembed(wordvocabsize, tagvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,
                    batch_size=32):

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=False,
                                    weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.25)(word_embedding_sent)

    char_input_sent = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_sent)

    char_cnn_sent = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_sent = char_cnn_sent(char_embedding_sent)
    char_embedding_sent = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent)
    char_embedding_sent = Dropout(0.25)(char_embedding_sent)

    input_e1_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e1_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e1_posi)

    input_e2_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e2_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e2_posi)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)

    embedding_x1 = concatenate([word_embedding_sent, char_embedding_sent, embedding_e1_posi, embedding_e2_posi], axis=-1)
    BiLSTM_x1 = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # attention_probs = Activation('softmax')(attention_concat)
    attention_probs = Dense(1, activation='softmax')(attention_self)
    # attention_probs = Flatten()(attention_probs)
    # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    attention_x1 = Dropout(0.25)(representation)

    mlp_x2_0 = Flatten()(tag_embedding)
    mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    distance = dot([mlp_x1_2, mlp_x2_0], axes=-1, normalize=True)

    mymodel = Model([word_input_sent, input_e1_posi, input_e2_posi, input_tag, char_input_sent], distance)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM_sent__MLP_KGembed(wordvocabsize, tagvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,
                    batch_size=32):

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=True,
                                    trainable=False,
                                    weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.25)(word_embedding_sent)

    char_input_sent = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_sent)

    char_cnn_sent = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_sent = char_cnn_sent(char_embedding_sent)
    char_embedding_sent = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent)
    char_embedding_sent = Dropout(0.25)(char_embedding_sent)

    input_e1_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e1_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e1_posi)

    input_e2_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    embedding_e2_posi = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])(input_e2_posi)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)

    embedding_x1 = concatenate([word_embedding_sent, char_embedding_sent, embedding_e1_posi, embedding_e2_posi], axis=-1)
    BiLSTM_x1 = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # attention_probs = Activation('softmax')(attention_concat)
    attention_probs = Dense(1, activation='softmax')(attention_self)
    # attention_probs = Flatten()(attention_probs)
    # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    attention_x1 = Dropout(0.25)(representation)

    mlp_x2_0 = Flatten()(tag_embedding)
    mlp_x2_1 = Dense(200, activation='tanh')(mlp_x2_0)
    mlp_x2_1 = Dropout(0.25)(mlp_x2_1)
    mlp_x2_2 = Dense(400, activation='tanh')(mlp_x2_1)
    mlp_x2_2 = Dropout(0.25)(mlp_x2_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    distance = dot([attention_x1, mlp_x2_2], axes=-1, normalize=True)

    mymodel = Model([word_input_sent, input_e1_posi, input_e2_posi, input_tag, char_input_sent], distance)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM__MLP_context(wordvocabsize, tagvocabsize, posivocabsize,
                     word_W, posi_W, tag_W,
                     input_sent_lenth, input_frament_lenth,
                     w2v_k, posi2v_k, tag2v_k,
                    batch_size=32):

    word_input_f = Input(shape=(input_frament_lenth,), dtype='int32')
    word_embedding_f = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_frament_lenth,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[word_W])(word_input_f)
    # word_embedding_f = Dropout(0.5)(word_embedding_f)

    word_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_l = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_l)
    # word_embedding_context_l = Dropout(0.5)(word_input_context_l)

    word_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_r = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_r)
    # word_embedding_context_r = Dropout(0.5)(word_embedding_context_r)

    posi_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_l = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_l)
    # posi_embedding_context_l = Dropout(0.5)(posi_embedding_context_l)

    posi_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_r = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_r)
    # posi_embedding_context_r = Dropout(0.5)(posi_embedding_context_r)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[tag_W])(input_tag)



    cnn2 = Conv1D(100, 2, activation='relu', strides=1, padding='valid')(word_embedding_f)
    cnn2 = Dropout(0.3)(cnn2)
    cnn4 = Conv1D(100, 4, activation='relu', strides=1, padding='valid')(cnn2)
    CNN_x1_f = GlobalMaxPooling1D()(cnn4)

    embedding_x1_l = concatenate([word_embedding_context_l, posi_embedding_context_l], axis=-1)
    BiLSTM_x1_l = LSTM(100, activation='tanh',return_sequences=False)(embedding_x1_l)

    embedding_x1_r = concatenate([word_embedding_context_r, posi_embedding_context_r], axis=-1)
    BiLSTM_x1_r = LSTM(100, activation='tanh',return_sequences=False, go_backwards=True)(embedding_x1_r)

    x1_all = concatenate([BiLSTM_x1_l, CNN_x1_f, BiLSTM_x1_r], axis=-1)
    x1_all = Dropout(0.5)(x1_all)

    mlp_x2_0 = Flatten()(tag_embedding)
    mlp_x2_0 = Dropout(0.5)(mlp_x2_0)
    mlp_x2_1_1 = Dense(400, activation='tanh')(mlp_x2_0)
    mlp_x2_1_1 = Dropout(0.5)(mlp_x2_1_1)
    # mlp_x2_1_2 = Dense(200, activation='relu')(mlp_x2_0)
    # mlp_x2_1_2 = Dropout(0.5)(mlp_x2_1_2)
    # mlp_x2_1 = concatenate([mlp_x2_1_1, mlp_x2_1_2])
    mlp_x2_2 = Dense(300, activation='tanh')(mlp_x2_1_1)
    x2_all = Dropout(0.5)(mlp_x2_2)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([x1_all, x2_all])

    mymodel = Model([word_input_context_l, posi_input_context_l,
                     word_input_context_r, posi_input_context_r,
                     word_input_f, input_tag], distance)

    mymodel.compile(loss=contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM__MLP_context_withClassifer(wordvocabsize, tagvocabsize, posivocabsize,
                     word_W, posi_W, tag_W,
                     input_sent_lenth, input_frament_lenth,
                     w2v_k, posi2v_k, tag2v_k,
                    batch_size=32):

    word_input_f = Input(shape=(input_frament_lenth,), dtype='int32')
    word_embedding_f = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_frament_lenth,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[word_W])(word_input_f)
    # word_embedding_f = Dropout(0.5)(word_embedding_f)

    word_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_l = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_l)
    # word_embedding_context_l = Dropout(0.5)(word_input_context_l)

    word_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_r = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_r)
    # word_embedding_context_r = Dropout(0.5)(word_embedding_context_r)

    posi_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_l = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_l)
    # posi_embedding_context_l = Dropout(0.5)(posi_embedding_context_l)

    posi_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_r = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_r)
    # posi_embedding_context_r = Dropout(0.5)(posi_embedding_context_r)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[tag_W])(input_tag)



    cnn2 = Conv1D(100, 2, activation='relu', strides=1, padding='valid')(word_embedding_f)
    cnn2 = Dropout(0.3)(cnn2)
    cnn4 = Conv1D(100, 4, activation='relu', strides=1, padding='valid')(cnn2)
    CNN_x1_f = GlobalMaxPooling1D()(cnn4)

    embedding_x1_l = concatenate([word_embedding_context_l, posi_embedding_context_l], axis=-1)
    BiLSTM_x1_l = LSTM(100, activation='tanh',return_sequences=False)(embedding_x1_l)

    embedding_x1_r = concatenate([word_embedding_context_r, posi_embedding_context_r], axis=-1)
    BiLSTM_x1_r = LSTM(100, activation='tanh',return_sequences=False, go_backwards=True)(embedding_x1_r)

    x1_all = concatenate([BiLSTM_x1_l, CNN_x1_f, BiLSTM_x1_r], axis=-1)
    x1_all = Dropout(0.5)(x1_all)

    # classifer = Dense(tagvocabsize, activation='softmax', name='classifer')(x1_all)


    mlp_x2_0 = Flatten()(tag_embedding)
    mlp_x2_0 = Dropout(0.5)(mlp_x2_0)
    mlp_x2_1_1 = Dense(400, activation='tanh')(mlp_x2_0)
    mlp_x2_1_1 = Dropout(0.5)(mlp_x2_1_1)
    # mlp_x2_1_2 = Dense(200, activation='relu')(mlp_x2_0)
    # mlp_x2_1_2 = Dropout(0.5)(mlp_x2_1_2)
    # mlp_x2_1 = concatenate([mlp_x2_1_1, mlp_x2_1_2])
    mlp_x2_2 = Dense(300, activation='tanh')(mlp_x2_1_1)
    x2_all = Dropout(0.5)(mlp_x2_2)


    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape, name='edistance')([x1_all, x2_all])

    classifer = Dense(2, activation='softmax', name='classifer')(distance)

    mymodel = Model([word_input_context_l, posi_input_context_l,
                     word_input_context_r, posi_input_context_r,
                     word_input_f, input_tag], [classifer])

    mymodel.compile(loss={'classifer': 'categorical_crossentropy'},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'classifer': ['acc']})
    # mymodel.compile(loss={'classifer': 'categorical_crossentropy', 'edistance': contrastive_loss},
    #                 loss_weights={'classifer': 1., 'edistance': 1.},
    #                 optimizer=optimizers.Adam(lr=0.001),
    #                 metrics={'classifer': ['acc'], 'edistance': [acc_siamese]})

    return mymodel



def Model_BiLSTM__MLP_context_withClassifer_charembedTYPE(wordvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, char_W,
                     input_sent_lenth, input_frament_lenth,
                     w2v_k, posi2v_k, c2v_k, tag_k,
                    batch_size=32):
    input_maxword_length = 18


    word_input_f = Input(shape=(input_frament_lenth,), dtype='int32')
    word_embedding_f = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_frament_lenth,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[word_W])(word_input_f)
    # word_embedding_f = Dropout(0.5)(word_embedding_f)

    char_input_f = Input(shape=(input_frament_lenth, input_maxword_length,), dtype='int32')
    char_embedding_f = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_frament_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_f)

    char_cnn_f = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_f(char_embedding_f)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_f = Dropout(0.25)(char_embedding_fragment)

    embedding_f = concatenate([word_embedding_f, char_embedding_f], axis=-1)


    word_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_l = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_l)
    # word_embedding_context_l = Dropout(0.5)(word_input_context_l)

    char_input_context_l = Input(shape=(input_sent_lenth+1, input_maxword_length,), dtype='int32')
    char_embedding_context_l = TimeDistributed(Embedding(input_dim=charvocabsize,
                                                           output_dim=c2v_k,
                                                           batch_input_shape=(
                                                               batch_size, input_sent_lenth+1,
                                                               input_maxword_length),
                                                           mask_zero=False,
                                                           trainable=True,
                                                           weights=[char_W]))(char_input_context_l)
    char_cnn_c = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_leftcontext = char_cnn_c(char_embedding_context_l)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_context_l = Dropout(0.25)(char_embedding_leftcontext)

    word_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    word_embedding_context_r = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])(word_input_context_r)
    # word_embedding_context_r = Dropout(0.5)(word_embedding_context_r)

    char_input_context_r = Input(shape=(input_sent_lenth+1, input_maxword_length,), dtype='int32')
    char_embedding_context_r = TimeDistributed(Embedding(input_dim=charvocabsize,
                                                            output_dim=c2v_k,
                                                            batch_input_shape=(
                                                                batch_size, input_sent_lenth+1,
                                                                input_maxword_length),
                                                            mask_zero=False,
                                                            trainable=True,
                                                            weights=[char_W]))(char_input_context_r)
    char_embedding_rightcontext = char_cnn_c(char_embedding_context_r)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_context_r = Dropout(0.25)(char_embedding_rightcontext)


    posi_input_context_l = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_l = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_l)
    # posi_embedding_context_l = Dropout(0.5)(posi_embedding_context_l)

    posi_input_context_r = Input(shape=(input_sent_lenth+1,), dtype='int32')
    posi_embedding_context_r = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth+1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[posi_W])(posi_input_context_r)
    # posi_embedding_context_r = Dropout(0.5)(posi_embedding_context_r)

    input_tag = Input(shape=(tag_k,), dtype='int32')
    tag_embedding = Embedding(input_dim=charvocabsize,
                                    output_dim=c2v_k,
                                    input_length=tag_k,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[char_W])(input_tag)



    cnn2 = Conv1D(100, 2, activation='relu', strides=1, padding='valid')(embedding_f)
    cnn2 = Dropout(0.3)(cnn2)
    cnn4 = Conv1D(100, 4, activation='relu', strides=1, padding='valid')(cnn2)
    CNN_x1_f = GlobalMaxPooling1D()(cnn4)

    embedding_x1_l = concatenate([word_embedding_context_l, char_embedding_context_l, posi_embedding_context_l], axis=-1)
    BiLSTM_x1_l = LSTM(100, activation='tanh',return_sequences=False)(embedding_x1_l)

    embedding_x1_r = concatenate([word_embedding_context_r, char_embedding_context_r, posi_embedding_context_r], axis=-1)
    BiLSTM_x1_r = LSTM(100, activation='tanh',return_sequences=False, go_backwards=True)(embedding_x1_r)

    x1_all = concatenate([BiLSTM_x1_l, CNN_x1_f, BiLSTM_x1_r], axis=-1)
    x1_all = Dropout(0.5)(x1_all)

    # classifer = Dense(tagvocabsize, activation='softmax', name='classifer')(x1_all)


    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x2_0 = Dropout(0.5)(mlp_x2_0)
    # mlp_x2_1_1 = Dense(400, activation='tanh')(mlp_x2_0)
    # mlp_x2_1_1 = Dropout(0.5)(mlp_x2_1_1)
    # # mlp_x2_1_2 = Dense(200, activation='relu')(mlp_x2_0)
    # # mlp_x2_1_2 = Dropout(0.5)(mlp_x2_1_2)
    # # mlp_x2_1 = concatenate([mlp_x2_1_1, mlp_x2_1_2])
    # mlp_x2_2 = Dense(300, activation='tanh')(mlp_x2_1_1)
    # x2_all = Dropout(0.5)(mlp_x2_2)

    BiLSTM_x2 = Bidirectional(LSTM(150, activation='tanh', return_sequences=False))(tag_embedding)
    x2_all = Dropout(0.5)(BiLSTM_x2)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape, name='edistance')([x1_all, x2_all])

    classifer = Dense(2, activation='softmax', name='classifer')(distance)

    mymodel = Model([word_input_context_l, posi_input_context_l, char_input_context_l,
                     word_input_context_r, posi_input_context_r, char_input_context_r,
                     word_input_f, char_input_f, input_tag], [classifer])

    mymodel.compile(loss={'classifer': 'categorical_crossentropy'},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'classifer': ['acc']})
    # mymodel.compile(loss={'classifer': 'categorical_crossentropy', 'edistance': contrastive_loss},
    #                 loss_weights={'classifer': 1., 'edistance': 1.},
    #                 optimizer=optimizers.Adam(lr=0.001),
    #                 metrics={'classifer': ['acc'], 'edistance': [acc_siamese]})

    return mymodel





def euclidean_distance(vects):
    # 计算欧式距离
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    # 在这里我们需要求修改output_shape, 为(batch, 1)
    shape1, shape2 = shapes
    return (shape1[0], 1)


# 创建训练时计算acc的方法
def acc_siamese(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def anti_contrastive_loss(y_true, y_pred):

    margin = 0.5
    return K.mean((1 - y_true) * K.square(y_pred) +
                  y_true * K.square(1 - y_pred))

def Model3_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):
    hidden_dim = 100

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')

    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = word_embedding_leftcontext
    embedding_rightcontext = word_embedding_rightcontext

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)
    Rep_LSTM_leftcontext = RepeatVector(input_fragment_lenth)(LSTM_leftcontext)
    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)
    Rep_LSTM_rightcontext = RepeatVector(input_fragment_lenth)(LSTM_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh',return_sequences=True), merge_mode='concat')(embedding_fragment)
    context_ADD = add([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])
    context_subtract_l = subtract([BiLSTM_fragment, LSTM_leftcontext])
    context_subtract_r = subtract([BiLSTM_fragment, LSTM_rightcontext])
    context_average = average([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])
    context_maximum = maximum([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])

    embedding_mix = concatenate([embedding_fragment, BiLSTM_fragment,
                                 context_ADD, context_subtract_l, context_subtract_r,
                                 context_average, context_maximum], axis=-1)

    # BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    decoderlayer1 = Conv1D(50, 1, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer3 = Conv1D(50, 3, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_mix)

    CNNs_fragment = concatenate([decoderlayer1, decoderlayer2, decoderlayer3, decoderlayer4], axis=-1)
    CNNs_fragment = Dropout(0.5)(CNNs_fragment)
    CNNs_fragment = GlobalMaxPooling1D()(CNNs_fragment)

    concat = Dropout(0.3)(CNNs_fragment)


    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models


def Model_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)


    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.5)(concat)
    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models


def Model_LSTM_BiLSTM_LSTM_simul(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)

    LSTM_leftcontext = Bidirectional(LSTM(hidden_dim, go_backwards=False, activation='tanh'), merge_mode='ave')(embedding_leftcontext)

    LSTM_rightcontext = Bidirectional(LSTM(hidden_dim, go_backwards=True, activation='tanh'), merge_mode='ave')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)


    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.5)(concat)

    output_2t = Dense(2, activation='softmax', name='2type')(concat)

    output_2t_2input = Dense(100, activation=None)(output_2t)

    concat2 = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext, output_2t_2input], axis=-1)
    concat2 = Dropout(0.5)(concat2)

    output = Dense(targetvocabsize, activation='softmax', name='5type')(concat2)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], [output_2t, output])

    Models.compile(loss='categorical_crossentropy',
                   loss_weights={'5type': 1., '2type': 0.4},
                   optimizer=optimizers.RMSprop(lr=0.001),
                   metrics=['acc'])

    return Models


def Model_3Level(wordvocabsize, targetvocabsize, charvobsize, posivocabsize,
                     word_W, char_W, posi_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth,
                     input_maxword_length, input_sent_lenth,
                     w2v_k, c2v_k, posi_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):



    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.5)(word_embedding_sent)

    word_input_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_posi = Embedding(input_dim=posivocabsize,
                               output_dim=posi_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[posi_W])(word_input_posi)
    word_embedding_posi = Dropout(0.5)(word_embedding_posi)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)
    embedding_posi = Dense(50, activation=None)(word_embedding_posi)
    embedding_sent = concatenate([word_embedding_sent, embedding_posi], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    BiLSTM_sent = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_sent)

    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext, BiLSTM_sent], axis=-1)
    concat = Dropout(0.5)(concat)
    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    word_input_posi, word_input_sent,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models

def Model_3Level_tag2v(wordvocabsize, targetvocabsize, charvobsize, posivocabsize,
                     word_W, char_W, posi_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth,
                     input_maxword_length, input_sent_lenth,
                     w2v_k, c2v_k, posi_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):



    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.5)(word_embedding_sent)

    word_input_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_posi = Embedding(input_dim=posivocabsize,
                               output_dim=posi_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[posi_W])(word_input_posi)
    word_embedding_posi = Dropout(0.5)(word_embedding_posi)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)
    embedding_posi = Dense(50, activation=None)(word_embedding_posi)
    embedding_sent = concatenate([word_embedding_sent, embedding_posi], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    BiLSTM_sent = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(embedding_sent)

    tag2vec_input = Input(shape=(5, ), dtype='float32')
    tag2vec_dense = Dense(200 * 2, activation='tanh')(tag2vec_input)

    # Manhattan = subtract([BiLSTM_sent, tag2vec_dense])
    # Manhattan = Lambda(lambda x: K.abs(x)))(Manhattan)
    # Manhattan_distance = merge([BiLSTM_sent, tag2vec_dense], mode=lambda x: Get_Manhattan(x[0], x[1]),
    #                            output_shape=lambda x: (x[0][0], 1))

    distance = Lambda(Manhattan_distance, output_shape=eucl_dist_output_shape)([BiLSTM_sent, tag2vec_dense])

    output = Dense(2, activation='softmax')(distance)

    # concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext, BiLSTM_sent], axis=-1)
    # concat = Dropout(0.5)(concat)
    # output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    word_input_posi, word_input_sent,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext,
                    tag2vec_input], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models


def Manhattan_distance(vects):
    left, right = vects
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def Euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        # self.supports_masking = True
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        # config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

