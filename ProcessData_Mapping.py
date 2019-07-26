#coding=utf-8
__author__ = 'JIA'


import numpy as np
import pickle, codecs
import json
import re, random, math
import keras
from Seq2fragment import Seq2frag, Seq2frag4test


def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W


def load_vec_txt(fname, vocab, k=300):
    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f.readlines():
        values = line.rstrip('\n').split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        lower_word = word.lower()
        if not w2v.__contains__(lower_word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[lower_word]

    print('UnKnown tokens in w2v', unknowtoken)
    return w2v,k,W


def load_vec_KGrepresentation(fname, vocab, k):

    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    for line in f.readlines():
        values = line.rstrip('\n').split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()

    W = np.zeros(shape=(vocab.__len__(), k))
    for item in vocab:

        try:
            W[vocab[item]] = w2v[item]
        except BaseException:
            print('the rel is not finded ...')

    return k, W


def load_vec_Sentrepresentation(s2v_k, s2v):

    W = np.zeros(shape=(s2v.__len__(), s2v_k))
    for its in s2v.keys():
        W[its] = s2v[its]

    return s2v_k, W


def get_sent_index(s2v_file, s2v, start=0):

    tag2sent_Dict = {}

    f = codecs.open(s2v_file, 'r', encoding='utf-8')

    for line in f.readlines():
        values = line.rstrip('\n').split()
        sent = (int(values[0])+start, int(values[1]))
        coefs = np.asarray(values[2:], dtype='float32')
        assert sent[0] not in s2v.keys()
        s2v[sent[0]] = coefs

        if sent[1] not in tag2sent_Dict.keys():
            tag2sent_Dict[sent[1]] = []
        tag2sent_Dict[sent[1]] += [sent[0]]

    f.close()

    return s2v, tag2sent_Dict


def load_vec_random(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__(), k))

    for i in vocab_c_inx.keys():
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return k, W


def load_vec_onehot(k=124):
    vocab_w_inx = [i for i in range(0, k)]

    W = np.zeros(shape=(vocab_w_inx.__len__(), k))

    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.

    return k, W


def CreatePairs_sentpair(tagDict_train):

    labels = []
    data_s_all = []
    data_t_all = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]


        for s in range(1,len(sents)):
            labels.append(1)
            data_s_all.append([sents[s]])
            data_t_all.append([sents[0]])

            labels.append(0)
            data_s_all.append([sents[s]])
            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            data_t_all.append([tagDict_train[keylist[ran1]][0]])


    pairs = [data_s_all, data_t_all]

    return pairs, labels


def CreatePairs(tagDict_train):

    labels = []
    data_s_all = []
    data_t_all = []

    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        for sent in sents:
            labels.append(1)
            data_s_all.append([sent])
            data_t_all.append([tag])

            labels.append(0)
            data_s_all.append([sent])
            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            data_t_all.append([keylist[ran1]])


    pairs = [data_s_all, data_t_all]

    return pairs, labels


def get_data(sentpair_datafile, s2v_trainfile, s2v_testfile, t2v_file, datafile, s2v_k=400, t2v_k=100):

    """
    数据处理的入口函数
    Converts the input files  into the model input formats
    """
    tagDict_train, tagDict_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    char_vob, char_id2char, char_W, c2v_k,\
    target_vob, target_id2word, type_W, type_k,\
    posi_W, posi_k,\
    max_s, max_posi, max_c = pickle.load(open(sentpair_datafile, 'rb'))

    s2v_dict = {}
    s2v_dict, tag2sentDict_train = get_sent_index(s2v_trainfile, s2v_dict)
    print('s2v_dict, tag2sentDict_train len', len(s2v_dict), len(tag2sentDict_train))

    s2v_dict, tag2sentDict_test = get_sent_index(s2v_testfile, s2v_dict, start=len(s2v_dict))
    print('s2v_dict, tag2sentDict_test len', len(s2v_dict), len(tag2sentDict_test))

    sent_k, sent_W = load_vec_Sentrepresentation(s2v_k, s2v_dict)
    print('sent_k, sent_W len', sent_k, len(sent_W))

    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=t2v_k)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))

    # pairs_train, labels_train = CreatePairs(tag2sentDict_train)
    # print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))
    #
    # pairs_test, labels_test = CreatePairs(tag2sentDict_test)
    # print('CreatePairs test len = ', len(pairs_test[0]), len(labels_test))


    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([tag2sentDict_train, tag2sentDict_test,
                 sent_W, sent_k,
                 target_vob, target_id2word, type_W, type_k], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    alpha = 10
    maxlen = 50

    t2v_file = './data/KG2v/FB15K_PTransE_Relation2Vec_100.txt'
    s2v_trainfile = './data/Model_BiLSTM_SentPair_1__data_Siamese.WordChar.Sentpair__1.h5.train.txt'
    s2v_testfile = './data/Model_BiLSTM_SentPair_1__data_Siamese.WordChar.Sentpair__1.h5.test.txt'
    resultdir = "./data/result/"

    # datafname = 'data_Siamese.4_allneg' #1,3, 4_allneg, 4_allneg_segmentNeg
    datafname = 'data_Mapping.PTransE'

    datafile = "./model/model_data/" + datafname + ".pkl"

    sentpair_datafile = "./model/model_data/data_Siamese.WordChar.Sentpair.pkl"

    tagDict_train, tagDict_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    char_vob, char_id2char, char_W, c2v_k,\
    target_vob, target_id2word, type_W, type_k,\
    posi_W, posi_k,\
    max_s, max_posi, max_c = pickle.load(open(sentpair_datafile, 'rb'))
    s2v_k = 400
    t2v_k = 100


    sent_W = {}
    sent_k, sent_W, tag2sentDict_train = load_vec_Sentrepresentation(s2v_trainfile, s2v_k, sent_W)
    print('sent_k, sent_W, tag2sentDict_train len', sent_k, len(sent_W), len(tag2sentDict_train))

    sent_k, sent_W, tag2sentDict_test = load_vec_Sentrepresentation(s2v_testfile, s2v_k, sent_W, start=len(sent_W))
    print('sent_k, sent_W, tag2sentDict_test len', sent_k, len(sent_W), len(tag2sentDict_test))

    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=t2v_k)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))

    pairs_train, labels_train = CreatePairs(tag2sentDict_train)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))

    pairs_test, labels_test = CreatePairs(tag2sentDict_test)
    print('CreatePairs test len = ', len(pairs_test[0]), len(labels_test))