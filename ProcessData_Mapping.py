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


def load_vec_Sentrepresentation(s2v_file, s2v_k, s2v, start=0):

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

    return s2v_k, s2v, tag2sent_Dict


def load_vec_random(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__(), k))

    for i in vocab_c_inx.keys():
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return k, W


def load_vec_Charembed(vocab_c_inx, char_vob,  k=30):

    TYPE = ['location', 'organization', 'person', 'miscellaneous']

    max = 13
    W = {}

    for i, tystr in enumerate(TYPE):
        for ch in tystr:
            if i not in W.keys():
                W[i] = [char_vob[ch]]
            else:
                W[i] += [char_vob[ch]]

        W[i] += [0 for s in range(max-len(tystr))]

    return max, W


def load_vec_character(c2vfile, vocab_c_inx, k=50):

    fi = open(c2vfile, 'r')
    c2v = {}
    for line in fi:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        c2v[word] = coefs
    fi.close()

    c2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        if not c2v.__contains__(i):
            c2v[i] = c2v["**UNK**"]
            W[vocab_c_inx[i]] = c2v[i]
        else:
            W[vocab_c_inx[i]] = c2v[i]

    return W, k


def load_vec_onehot(k=124):
    vocab_w_inx = [i for i in range(0, k)]

    W = np.zeros(shape=(vocab_w_inx.__len__(), k))

    for word in vocab_w_inx:
        W[vocab_w_inx[word], vocab_w_inx[word]] = 1.

    return k, W


def make_idx_word_index(file, max_s, source_vob, target_vob):

    data_s_all = []
    data_t_all = []
    data_tO_all = []
    data_tBIOES_all = []
    data_tType_all = []

    f = open(file, 'r')
    fr = f.readlines()

    count = 0
    data_t = []
    data_tO = []
    data_tBIOES = []
    data_tType = []
    data_s = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append(0)
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[0] = 1
                data_t.append(targetvec)

                targetvecO = np.zeros(2 + 1)
                targetvecO[0] = 1
                data_tO.append(targetvecO)

                # data_tO.append([0.00])

                targetvecBIOES = np.zeros(5 + 1)
                targetvecBIOES[0] = 1
                data_tBIOES.append(targetvecBIOES)

                targetvecType = np.zeros(5 + 1)
                targetvecType[0] = 1
                data_tType.append(targetvecType)

            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_t_all.append(data_t)
            data_tO_all.append(data_tO)
            data_tBIOES_all.append(data_tBIOES)
            data_tType_all.append(data_tType)
            data_t = []
            data_tO = []
            data_tBIOES = []
            data_tType = []
            data_s = []
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])

        # data_t.append(target_vob[sent[4]])
        targetvec = np.zeros(len(target_vob) + 1)
        targetvec[target_vob[sent[4]]] = 1
        data_t.append(targetvec)

        targetvecO = np.zeros(2 + 1)
        if sent[4] == 'O':
            targetvecO[1] = 1
        else:
            targetvecO[2] = 1
        data_tO.append(targetvecO)


        targetvecBIOES = np.zeros(5 + 1)
        if sent[4] == 'O':
            targetvecBIOES[3] = 1
        elif sent[4][0] == 'B':
            targetvecBIOES[1] = 1
        elif sent[4][0] == 'I':
            targetvecBIOES[2] = 1
        elif sent[4][0] == 'E':
            targetvecBIOES[4] = 1
        elif sent[4][0] == 'S':
            targetvecBIOES[5] = 1
        else:
            targetvecBIOES[5] = 1
        data_tBIOES.append(targetvecBIOES)

        targetvecType = np.zeros(5 + 1)

        if sent[4] == 'O':
            targetvecType[1] = 1
        elif 'LOC' in sent[4]:
            targetvecType[2] = 1
        elif 'ORG' in sent[4]:
            targetvecType[3] = 1
        elif 'PER' in sent[4]:
            targetvecType[4] = 1
        elif 'MISC' in sent[4]:
            targetvecType[5] = 1
        else:
            targetvecType[1] = 1
        data_tType.append(targetvecType)

        count += 1

    f.close()

    return [data_s_all, data_t_all, data_tO_all, data_tBIOES_all, data_tType_all]


def make_idx_character_index(file, max_s, max_c, source_vob):

    data_s_all=[]
    count = 0
    f = open(file,'r')
    fr = f.readlines()

    data_w = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)

            for inum in range(0, num):
                data_tmp = []
                for i in range(0, max_c):
                    data_tmp.append(0)
                data_w.append(data_tmp)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_w)

            data_w = []
            count =0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]

        for chr in range(0, min(word.__len__(), max_c)):
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])

        num = max_c - word.__len__()
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count +=1
        data_w.append(data_c)

    f.close()
    return data_s_all


def get_word_index(files):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount = 0

    max_s = 0

    if not source_vob.__contains__("**PlaceHolder**"):
        source_vob["**PlaceHolder**"] = count
        sourc_idex_word[count] = "**PlaceHolder**"
        count += 1
    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    for testf in files:

        f = codecs.open(testf, 'r', encoding='utf-8')
        for line in f.readlines():

            jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
            sent = jline['sent']
            rel = jline['rel']
            words = sent.split(' ')
            for word in words:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1
            if not target_vob.__contains__(rel):
                target_vob[rel] = tarcount
                target_idex_word[tarcount] = rel
                tarcount += 1

            max_s = max(max_s, len(words))

        f.close()

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def get_Character_index(files):


    source_vob = {}
    sourc_idex_word = {}
    max_c = 18
    count = 1

    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**Placeholder**"):
        source_vob["**Placeholder**"] = 1
        sourc_idex_word[1] = "**Placeholder**"
        count += 1

    for file in files:

        f = codecs.open(file, 'r', encoding='utf-8')
        for line in f.readlines():
            jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
            sent = jline['sent']
            rel = jline['rel']
            words = sent.split(' ')

            for word in words:
                for character in word:
                    if not source_vob.__contains__(character):
                        source_vob[character] = count
                        sourc_idex_word[count] = character
                        count += 1

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_c


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
            data_t_all.append(keylist[ran1])


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

    sent_W = {}
    sent_k, sent_W, tag2sentDict_train = load_vec_Sentrepresentation(s2v_trainfile, s2v_k, sent_W)
    print('sent_k, sent_W, tag2sentDict_train len', sent_k, len(sent_W), len(tag2sentDict_train))

    sent_k, sent_W, tag2sentDict_test = load_vec_Sentrepresentation(s2v_testfile, s2v_k, sent_W, start=len(sent_W))
    print('sent_k, sent_W, tag2sentDict_test len', sent_k, len(sent_W), len(tag2sentDict_test))

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