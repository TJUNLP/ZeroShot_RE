# coding:utf-8

from keras.layers.core import Dropout,RepeatVector, Reshape
from keras.layers.merge import concatenate, add, subtract, average, maximum,multiply
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Activation
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.layers import merge, Lambda,Flatten
from keras.layers.merge import dot


def Model_sent2tag_MLP_1(sentvocabsize, tagvocabsize,
                     sent_W, tag_W, s2v_k, tag2v_k):

    input_sent = Input(shape=(1,), dtype='int32')
    sent_embedding = Embedding(input_dim=sentvocabsize,
                                    output_dim=s2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[sent_W])(input_sent)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])(input_tag)

    x1_1 = Flatten()(sent_embedding)
    x2_0 = Flatten()(tag_embedding)

    # x1_1 = Dense(100, activation='tanh')(x1_0)

    sub = subtract([x2_0, x1_1])
    mul = multiply([x2_0, x1_1])
    max = maximum([x2_0, x1_1])
    avg = average([x2_0, x1_1])
    class_input = concatenate([tag_embedding, x1_1, sub, mul, max, avg], axis=-1)
    # class_input = Flatten()(class_input)
    class_mlp1 = Dense(200, activation='tanh')(class_input)
    class_mlp1 = Dropout(0.5)(class_mlp1)
    class_mlp2 = Dense(2)(class_mlp1)
    class_output = Activation('softmax', name='CLASS')(class_mlp2)

    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([mlp_x1_2, x2_0])
    # distance = dot([x1_0, x2_0], axes=-1, normalize=True)

    mymodel = Model([input_sent, input_tag], class_output)

    mymodel.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])

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
