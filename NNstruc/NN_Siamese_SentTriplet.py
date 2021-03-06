# coding:utf-8

from keras.layers.core import Dropout, RepeatVector, Reshape
from keras.layers.merge import concatenate, add, average, maximum, multiply
from keras.layers import subtract
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Lambda, Flatten, Activation
from keras.layers.merge import dot, Dot
from flipGradientTF import GradientReversal


def Model_BiLSTM_SentPair_tripletloss_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.1
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_triplet])

    return mymodel


def Model_BiLSTM_SentPair_tripletloss_01_(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel



def Model_BiLSTM_SentPair_tripletloss_Hloss_round(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.5
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    # loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 1.)) * K.relu(margin + X[0] - X[1]))([wrong_cos, right_cos])

    loss = Lambda(lambda X: K.relu(margin + X[0] - X[1]) + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1]))))([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_BiLSTM_SentPair_tripletloss_Hloss_exp05_at05(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.5
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 1.)) * K.relu(margin + X[0] - X[1]) + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * K.square(X[2]))([wrong_cos, right_cos, at_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001),metrics=[acc_triplet])

    return mymodel


def Model_BiLSTM_SentPair_tripletloss_Hloss_05_at_exp(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.5
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    loss = Lambda(lambda X: K.exp(K.relu(margin + X[0] - X[1]) / (margin + 1.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])))([wrong_cos, right_cos, at_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel



def Model_BiLSTM_SentPair_tripletloss_Hloss_05_at01_allexp_2m(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32, margin=0.5, at_margin=0.1):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)


    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    # margin = 0.5
    # margin = 0.1
    # at_margin = 0.1
    gamma = 2
    # + at_margin * K.round(K.maximum(0.5 - K.epsilon(), 0.5 + X[1] - X[0] - margin)) * (
            # K.square(K.relu(X[0] - 0.2)) + K.square(K.relu(0.8 - X[1])))
    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])))([wrong_cos, right_cos, at_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3], loss)

    mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    return mymodel


def Model_BiLSTM_SentPair_tripletloss_ed(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x1, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel



def Model_BiLSTM_SentPair_Atloss_ed_05(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    class_BiLSTM = Dropout(0.5)(class_BiLSTM)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x1, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_input = Dropout(0.5)(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])), name='TripletLoss')([wrong_cos, right_cos, at_cos])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_Atloss_ed_05_pinjie(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    # tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=True, return_state=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_seq_x1, forward_h_x1, forward_c, backward_h_x1, backward_c = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = concatenate([forward_h_x1, backward_h_x1], axis=-1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_seq_x2, forward_h_x2, forward_c, backward_h_x2, backward_c = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = concatenate([forward_h_x2, backward_h_x2], axis=-1)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_seq_x3, forward_h_x3, forward_c, backward_h_x3, backward_c = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = concatenate([forward_h_x3, backward_h_x3], axis=-1)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    # class_pinjie = concatenate([tag_embedding, BiLSTM_seq_x1], axis=1)
    class_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding, BiLSTM_seq_x1])

    class_BiLSTM = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(class_pinjie)
    class_BiLSTM = Dropout(0.5)(class_BiLSTM)
    # class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(class_BiLSTM)
    # class_BiLSTM = Dropout(0.5)(class_BiLSTM)

    class_mlp2 = Dense(2)(class_BiLSTM)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])), name='TripletLoss')([wrong_cos, right_cos, at_cos])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 0.5, 'CLASS': 1.},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_pinjie(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    # tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh', return_sequences=True, return_state=True), merge_mode='ave')


    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_seq_x1, forward_h_x1, forward_c, backward_h_x1, backward_c = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = concatenate([forward_h_x1, backward_h_x1], axis=-1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    # class_pinjie = concatenate([tag_embedding, BiLSTM_seq_x1], axis=1)
    class_pinjie = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([tag_embedding, BiLSTM_seq_x1])

    class_BiLSTM = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')(class_pinjie)
    class_BiLSTM = Dropout(0.5)(class_BiLSTM)
    # class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(class_BiLSTM)
    # class_BiLSTM = Dropout(0.5)(class_BiLSTM)

    class_mlp2 = Dense(2)(class_BiLSTM)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)



    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    class_output)

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
    return mymodel


def Model_BiLSTM_SentPair_Atloss_ed_05_nochar(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    class_BiLSTM = Dropout(0.5)(class_BiLSTM)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x1, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_input = Dropout(0.5)(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])
    at_cos = Dot(axes=-1, normalize=True, name='at_cos')([BiLSTM_x2, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    at_margin = 0.1
    gamma = 2

    loss = Lambda(lambda X: K.exp((margin + X[0] - X[1]) / (margin + 2.)) * (K.relu(margin + X[0] - X[1]) + at_margin * K.square(X[2])), name='TripletLoss')([wrong_cos, right_cos, at_cos])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_ed_1_01(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x1, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.1},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_ed_multi(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    _dot = dot([tag_embedding, class_BiLSTM], axes=-1, normalize=False)
    class_input = concatenate([tag_embedding, BiLSTM_x1, sub, mul, max, avg, _dot], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    classifer1 = Dense(1, activation=None)(class_mlp1)
    classifer2 = Dense(1, activation='sigmoid')(class_mlp1)
    classifer3 = Dense(1, activation='tanh')(class_mlp1)
    classifer4 = Dense(1, activation='relu')(class_mlp1)
    classifer5 = Dot(axes=-1, normalize=True)([tag_embedding, class_BiLSTM])
    classifer6 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([tag_embedding, class_BiLSTM])
    class_mlp2 = concatenate([classifer1, classifer2, classifer3,
                              classifer4, classifer5, classifer6], axis=-1)
    class_mlp2 = Dense(2)(class_mlp2)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_ed_withRank(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W, rank_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, rank2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)

    input_rank = Input(shape=(1,), dtype='int32')
    rank_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=rank2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[rank_W])(input_rank)
    rank_embedding = Flatten()(rank_embedding)

    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh',name='class_BiLSTM')(BiLSTM_x1)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, class_BiLSTM, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_input2 = concatenate([rank_embedding, class_mlp1], axis=-1)
    class_mlp2 = Dense(200, activation='tanh')(class_input2)
    class_mlp2 = Dropout(0.5)(class_mlp2)
    class_output = Dense(2)(class_mlp2)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3,
                     input_tag, input_rank],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_mse(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh', name='class_BiLSTM')(BiLSTM_x2)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x2, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_output = Dense(1, name='CLASS')(class_mlp1)


    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'mean_squared_error'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_nonL_withRank(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W, rank_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, rank2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)

    input_rank = Input(shape=(1,), dtype='int32')
    rank_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=rank2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[rank_W])(input_rank)
    rank_embedding = Flatten()(rank_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh', name='class_BiLSTM')(BiLSTM_x2)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x2, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_input2 = concatenate([rank_embedding, class_mlp1], axis=-1)
    class_mlp2 = Dense(200, activation='tanh')(class_input2)
    class_mlp2 = Dropout(0.5)(class_mlp2)
    class_output = Dense(1, activation='tanh', name='CLASS')(class_mlp2)


    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag, input_rank],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'mean_squared_error'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel



def Model_BiLSTM_SentPair_tripletloss_LR_withRank(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W, rank_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, rank2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)

    input_rank = Input(shape=(1,), dtype='int32')
    rank_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=rank2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[rank_W])(input_rank)
    rank_embedding = Flatten()(rank_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh', name='class_BiLSTM')(BiLSTM_x2)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x2, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_input2 = concatenate([rank_embedding, class_mlp1], axis=-1)
    class_mlp2 = Dense(200, activation='tanh')(class_input2)
    class_mlp2 = Dropout(0.5)(class_mlp2)
    class_output = Dense(1, activation='sigmoid', name='CLASS')(class_mlp2)


    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag, input_rank],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))
    # mean_squared_error
    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'binary_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_mse_withRank(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W, rank_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k, rank2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)

    input_rank = Input(shape=(1,), dtype='int32')
    rank_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=rank2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[rank_W])(input_rank)
    rank_embedding = Flatten()(rank_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(100, activation='tanh', name='class_BiLSTM')(BiLSTM_x2)
    sub = subtract([tag_embedding, class_BiLSTM])
    mul = multiply([tag_embedding, class_BiLSTM])
    max = maximum([tag_embedding, class_BiLSTM])
    avg = average([tag_embedding, class_BiLSTM])
    class_input = concatenate([tag_embedding, BiLSTM_x2, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_input2 = concatenate([rank_embedding, class_mlp1], axis=-1)
    class_mlp2 = Dense(200, activation='tanh')(class_input2)
    class_mlp2 = Dropout(0.5)(class_mlp2)
    class_output = Dense(1, name='CLASS')(class_mlp2)


    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag, input_rank],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'mean_squared_error'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
    return mymodel


def Model_BiLSTM_SentPair_tripletloss_softmax_ques(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
                     word_W, posi_W, char_W, tag_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k, tag2v_k,
                    batch_size=32):

    word_input_sent_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    word_input_sent_x3 = Input(shape=(input_sent_lenth,), dtype='int32')
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

    word_embedding_sent_x3 = word_embedding_sent_layer(word_input_sent_x3)
    word_embedding_sent_x3 = Dropout(0.25)(word_embedding_sent_x3)

    char_input_sent_x1 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x2 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_input_sent_x3 = Input(shape=(input_sent_lenth, input_maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(Embedding(input_dim=charvocabsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))

    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_embedding_sent_x2 = char_embedding_sent_layer(char_input_sent_x2)
    char_embedding_sent_x3 = char_embedding_sent_layer(char_input_sent_x3)

    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.25)(char_embedding_sent_x1)

    char_embedding_sent_x2 = char_cnn_sent_layer(char_embedding_sent_x2)
    char_embedding_sent_x2 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x2)
    char_embedding_sent_x2 = Dropout(0.25)(char_embedding_sent_x2)

    char_embedding_sent_x3 = char_cnn_sent_layer(char_embedding_sent_x3)
    char_embedding_sent_x3 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x3)
    char_embedding_sent_x3 = Dropout(0.25)(char_embedding_sent_x3)

    input_e1_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e1_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    input_e2_posi_x1 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x2 = Input(shape=(input_sent_lenth,), dtype='int32')
    input_e2_posi_x3 = Input(shape=(input_sent_lenth,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                    output_dim=posi2v_k,
                                    input_length=input_sent_lenth,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[posi_W])

    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e1_posi_x2 = embedding_posi_layer(input_e1_posi_x2)
    embedding_e1_posi_x3 = embedding_posi_layer(input_e1_posi_x3)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)
    embedding_e2_posi_x2 = embedding_posi_layer(input_e2_posi_x2)
    embedding_e2_posi_x3 = embedding_posi_layer(input_e2_posi_x3)

    # input_tag = Input(shape=(1,), dtype='int32')
    # tag_embedding = Embedding(input_dim=tagvocabsize,
    #                                 output_dim=tag2v_k,
    #                                 input_length=1,
    #                                 mask_zero=False,
    #                                 trainable=False,
    #                                 weights=[tag_W])(input_tag)
    # tag_embedding = Flatten()(tag_embedding)

    input_tag = Input(shape=(245,), dtype='int32')
    tag_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                    output_dim=w2v_k,
                                    input_length=245,
                                    mask_zero=True,
                                    trainable=True,
                                    weights=[word_W])
    tag_embedding = tag_embedding_layer(input_tag)
    tag_embedding = Dropout(0.25)(tag_embedding)

    tag_BiLSTM = Bidirectional(LSTM(100, activation='tanh'), merge_mode='concat')(tag_embedding)
    tag_BiLSTM = Dropout(0.5)(tag_BiLSTM)

    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    embedding_x3 = concatenate([word_embedding_sent_x3, char_embedding_sent_x3,
                                embedding_e1_posi_x3, embedding_e2_posi_x3], axis=-1)
    BiLSTM_x3 = BiLSTM_layer(embedding_x3)
    BiLSTM_x3 = Dropout(0.25)(BiLSTM_x3)

    class_BiLSTM = Dense(200, activation='tanh',name='class_BiLSTM')(BiLSTM_x2)
    sub = subtract([tag_BiLSTM, class_BiLSTM])
    mul = multiply([tag_BiLSTM, class_BiLSTM])
    max = maximum([tag_BiLSTM, class_BiLSTM])
    avg = average([tag_BiLSTM, class_BiLSTM])
    class_input = concatenate([tag_BiLSTM, class_BiLSTM, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([BiLSTM_x1, BiLSTM_x2])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([BiLSTM_x1, BiLSTM_x3])

    # margin = 1.
    margin = 0.5
    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]), name='TripletLoss')([wrong_cos, right_cos])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2,
                     word_input_sent_x3, input_e1_posi_x3, input_e2_posi_x3, char_input_sent_x3, input_tag],
                    [loss, class_output])

    # mymodel.compile(loss=lambda y_true,y_pred: y_pred, optimizer=optimizers.Adam(lr=0.001))

    mymodel.compile(loss={'TripletLoss': lambda y_true, y_pred: y_pred, 'CLASS': 'categorical_crossentropy'},
                    loss_weights={'TripletLoss': 1., 'CLASS': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics={'TripletLoss': [], 'CLASS': ['acc']})
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


def acc_triplet(y_true, y_pred):
    return K.mean(K.equal(0., K.maximum(0., y_pred)))



def Hierarchical_loss(X, margin=0.5, at_margin=0.1):

    loss = K.relu(margin + X[0] - X[1]) + \
           at_margin * K.round(0.5 + K.maximum(0, X[1][0]-X[0][0] - margin)) * \
           (K.square(K.maximum(X[0]-0.1, K.epsilon())) + K.square(K.maximum(0.9 - X[1], K.epsilon())))
    return loss


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def contrastive_softmax_loss(y_true, y_pred):

    return K.mean(y_true * y_pred + (1 - y_true) * (1 - y_pred))


def anti_contrastive_loss(y_true, y_pred):

    margin = 0.5
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(1 - y_pred))


def crude_anti_contrastive_loss(y_true, y_pred):

    margin0 = 0.2
    margin1 = 0.8
    return K.mean((1 - y_true) * K.square(K.maximum(y_pred - margin0, K.epsilon())) +
                  y_true * K.square(K.maximum(margin1 - y_pred, K.epsilon())))


def Mix_loss(y_true, y_pred, e=0.1):
    nb_classes = 100
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/100, y_pred)
    return (1-e)*loss1 + e*loss2


def cos_distance_loss(y_true, y_pred):

    # return K.sum(K.square(1 - y_pred), axis=1, keepdims=True)
    margin = 0.8
    return K.sum(K.square(K.maximum(margin - y_pred, K.epsilon())), axis=1, keepdims=True)


def Manhattan_distance(vects):
    left, right = vects
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def Euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))