# coding:utf-8

from keras.layers.core import Dropout, RepeatVector, Reshape
from keras.layers.merge import concatenate, add, subtract, average, maximum
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Lambda, Flatten, Activation
from keras.layers.merge import dot, Dot


def Model_BiLSTM_SentPair_RelPunish_3(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([BiLSTM_x1, BiLSTM_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(BiLSTM_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    # relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])
    # relPunish = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='relPunish')([sent_x1_mlp2, tag_embedding])
    relPunish = Lambda(lambda x: K.maximum(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True), K.epsilon()),
                       name='relPunish')([sent_x1_mlp2, tag_embedding])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    [output1, relPunish])

    mymodel.compile(loss={'Output1': anti_contrastive_loss, 'relPunish': lambda y_true, y_pred: y_pred},
                    loss_weights={'Output1': 1., 'relPunish': 0.3}, #0.2
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM_SentPair_RelPunish_2(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([BiLSTM_x1, BiLSTM_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(BiLSTM_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    [output1, relPunish])

    mymodel.compile(loss={'Output1': anti_contrastive_loss, 'relPunish': cos_distance_loss},
                    loss_weights={'Output1': 1., 'relPunish': 0.5},
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=[acc_siamese])

    return mymodel



def Model_BiLSTM_SentPair_RelPunish_1_crude(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([BiLSTM_x1, BiLSTM_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(BiLSTM_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    output1)

    mymodel.compile(loss=crude_anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    # mymodel.compile(loss={'Output1': anti_contrastive_loss, 'relPunish': }, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])


    return mymodel


def Model_BiLSTM_SentPair_RelPunish_1_atten(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)
    tag_embedding = Flatten()(tag_embedding)


    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh', return_sequences=True, return_state=True), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1_seq, BiLSTM_x1_lh, x1_lc, BiLSTM_x1_rh, x1_rc = BiLSTM_layer(embedding_x1)
    # BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1_seq)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2_seq, BiLSTM_x2_lh, x2_lc, BiLSTM_x2_rh, x2_rc  = BiLSTM_layer(embedding_x2)
    # BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2_seq)


    attention_tanh = Dense(1, activation='tanh')
    attention_probs = Activation('softmax')

    atten_x2_concat = concatenate([BiLSTM_x2_lh, BiLSTM_x2_rh], axis=-1)
    atten_x2_Repeat = RepeatVector(input_sent_lenth)(atten_x2_concat)
    atten_x1_concat = concatenate([BiLSTM_x1_seq, atten_x2_Repeat],axis=-1)
    attention_x1 = attention_tanh(atten_x1_concat)
    attention_probs_x1 = attention_probs(attention_x1)
    representation_x1 = Lambda(lambda x: x[0] * x[1])([BiLSTM_x1_seq, attention_probs_x1])
    representation_x1 = Dropout(0.25)(representation_x1)

    atten_x1_concat = concatenate([BiLSTM_x1_lh, BiLSTM_x1_rh],axis=-1)
    atten_x1_Repeat = RepeatVector(input_sent_lenth)(atten_x1_concat)
    atten_x2_concat = concatenate([BiLSTM_x2_seq, atten_x1_Repeat],axis=-1)
    attention_x2 = attention_tanh(atten_x2_concat)
    attention_probs_x2 = attention_probs(attention_x2)
    representation_x2 = Lambda(lambda x: x[0] * x[1])([BiLSTM_x2_seq, attention_probs_x2])
    representation_x2 = Dropout(0.25)(representation_x2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([representation_x1, representation_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(representation_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    output1)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    # mymodel.compile(loss={'Output1': anti_contrastive_loss, 'relPunish': }, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])


    return mymodel


def Model_BiLSTM_SentPair_RelPunish_1(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([BiLSTM_x1, BiLSTM_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(BiLSTM_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])


    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    output1)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    # mymodel.compile(loss={'Output1': anti_contrastive_loss, 'relPunish': }, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])


    return mymodel


def Model_BiLSTM_SentPair_3(wordvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, char_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    BiLSTM_layer = Bidirectional(LSTM(200, activation='tanh'), merge_mode='concat')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2], distance)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM_SentPair_1(wordvocabsize, posivocabsize, charvocabsize,
                     word_W, posi_W, char_W,
                     input_sent_lenth, input_maxword_length,
                     w2v_k, posi2v_k, c2v_k,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    BiLSTM_layer = Bidirectional(LSTM(100, activation='tanh'), merge_mode='ave')

    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.25)(BiLSTM_x1)

    embedding_x2 = concatenate([word_embedding_sent_x2, char_embedding_sent_x2,
                                embedding_e1_posi_x2, embedding_e2_posi_x2], axis=-1)
    BiLSTM_x2 = BiLSTM_layer(embedding_x2)
    BiLSTM_x2 = Dropout(0.25)(BiLSTM_x2)

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)
    #
    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2], distance)

    mymodel.compile(loss=anti_contrastive_loss, optimizer=optimizers.Adam(lr=0.001), metrics=[acc_siamese])

    return mymodel


def Model_BiLSTM_SentPair_RelPunish_4(wordvocabsize, posivocabsize, charvocabsize, tagvocabsize,
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
                               batch_input_shape=(batch_size, input_sent_lenth, input_maxword_length),
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

    # attention_self = Dense(1, activation='tanh')(BiLSTM_x1)
    # # attention_probs = Activation('softmax')(attention_concat)
    # attention_probs = Dense(1, activation='softmax')(attention_self)
    # # attention_probs = Flatten()(attention_probs)
    # # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    # representation = Lambda(lambda x: x[0] * x[1])([attention_probs, BiLSTM_x1])
    # attention_x1 = Dropout(0.25)(representation)

    # mlp_x2_0 = Flatten()(tag_embedding)
    # mlp_x1_1 = Dense(200, activation='tanh')(attention_x1)
    # mlp_x1_1 = Dropout(0.25)(mlp_x1_1)
    # mlp_x1_2 = Dense(100, activation='tanh')(mlp_x1_1)
    # # mlp_x1_2 = Dropout(0.25)(mlp_x1_2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([BiLSTM_x1, mlp_x2_2])
    # cos_distance = dot([BiLSTM_x1, BiLSTM_x2], axes=-1, normalize=True)
    output1 = Dot(axes=-1, normalize=True, name='Output1')([BiLSTM_x1, BiLSTM_x2])

    sent_x1_mlp1 = Dense(200, activation='tanh')(BiLSTM_x1)
    sent_x1_mlp1 = Dropout(0.25)(sent_x1_mlp1)
    sent_x1_mlp2 = Dense(100, activation='tanh')(sent_x1_mlp1)
    # relPunish = Dot(axes=-1, normalize=True, name='relPunish')([sent_x1_mlp2, tag_embedding])
    # relPunish = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='relPunish')([sent_x1_mlp2, tag_embedding])
    relPunish = Lambda(lambda x: K.maximum(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True), K.epsilon()),
                       name='relPunish')([sent_x1_mlp2, tag_embedding])

    mymodel = Model([word_input_sent_x1, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     word_input_sent_x2, input_e1_posi_x2, input_e2_posi_x2, char_input_sent_x2, input_tag],
                    [output1, relPunish])

    mymodel.compile(loss={'Output1': crude_anti_contrastive_loss, 'relPunish': lambda y_true, y_pred: y_pred},
                    loss_weights={'Output1': 1., 'relPunish': 0.2}, #0.2
                    optimizer=optimizers.Adam(lr=0.001),
                    metrics=[acc_siamese])
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
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(1 - y_pred))


def crude_anti_contrastive_loss(y_true, y_pred):

    margin0 = 0.2
    margin1 = 0.8
    return K.mean((1 - y_true) * K.square(K.maximum(y_pred - margin0, K.epsilon())) +
                  y_true * K.square(K.maximum(margin1 - y_pred, K.epsilon())))


def Mix_loss(y_true, y_pred):

    y_pred1 = y_pred[0]
    y_pred2 = y_pred[1]

    margin0 = 0.2
    margin1 = 0.8
    crude_anti_contrastive = K.mean((1 - y_true) * K.square(K.maximum(y_pred1 - margin0, K.epsilon())) +
                                    y_true * K.square(K.maximum(margin1 - y_pred1, K.epsilon())))
    lamd = 0.1
    return crude_anti_contrastive + lamd * y_pred2




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
