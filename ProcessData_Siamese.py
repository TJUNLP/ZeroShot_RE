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


def CreatePairs(trainfile, max_s, max_posi, word_vob, target_vob, type_W, char_vob, max_c, istest=False):

    labels = []
    data_s_all = []
    data_e1_posi_all = []
    data_e2_posi_all = []
    data_tag_all = []
    classifer_label = []
    char_s_all = []

    f = codecs.open(trainfile, 'r', encoding='utf-8')

    for line in f.readlines():
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_posi'][0]
        e1_r = jline['e1_posi'][1]
        e2_l = jline['e2_posi'][0]
        e2_r = jline['e2_posi'][1]

        data_tag = target_vob[rel]

        data_s = [word_vob[ww] for ww in sent[0:min(len(sent), max_s)]]+ [0] * max(0, max_s - len(sent))

        list_left = [min(i, max_posi) for i in range(1, e1_l+1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(e1_l, e1_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - e1_r + 1)]
        data_e1_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        list_left = [min(i, max_posi) for i in range(1, e2_l + 1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(e2_l, e2_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - e2_r + 1)]
        data_e2_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        char_s = []
        for word in sent:
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])
            data_c = data_c + [0] * max(max_c - word.__len__(), 0)
            char_s.append(data_c)
        char_s = char_s + [[0] * max_c] * max(0, max_s - len(char_s))


        data_s_all.append(data_s)
        data_tag_all.append([data_tag])
        data_e1_posi_all.append(data_e1_posi)
        data_e2_posi_all.append(data_e2_posi)
        labels.append(1)
        classifer_label.append(data_tag)
        char_s_all.append(char_s)

        if istest == False:

            inc = random.randrange(1, len(target_vob.keys()))
            dn = (data_tag + inc) % len(target_vob.keys())
            data_s_all.append(data_s)
            data_tag_all.append([dn])
            data_e1_posi_all.append(data_e1_posi)
            data_e2_posi_all.append(data_e2_posi)
            labels.append(0)
            classifer_label.append(data_tag)
            char_s_all.append(char_s)


    pairs = [data_s_all, data_tag_all,
             data_e1_posi_all, data_e2_posi_all, char_s_all]

    classifer_labels = keras.utils.to_categorical(classifer_label, len(target_vob))

    return pairs, labels, classifer_labels


def get_data(trainfile, testfile, w2v_file, c2v_file, t2v_file, datafile, w2v_k=300, c2v_k=25, t2v_k=100, maxlen = 50,
             hasNeg=False, percent=1):

    """
    数据处理的入口函数
    Converts the input files  into the model input formats
    """

    word_vob, word_id2word, target_vob, target_id2word, max_s = get_word_index({trainfile, testfile})
    print("source vocab size: ", str(len(word_vob)))
    print("word_id2word size: ", str(len(word_id2word)))
    print("target vocab size: " + str(len(target_vob)))
    print("target_id2word size: " + str(len(target_id2word)))
    # if max_s > maxlen:
    #     max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))


    char_vob, char_id2char, max_c = get_Character_index({trainfile, testfile})
    print("source char size: ", char_vob.__len__())
    print("max_c: ", max_c)
    print("source char: " + str(char_id2char))

    c2v_k, char_W, = load_vec_random(char_vob, k=c2v_k)
    print('character_W shape:', char_W.shape)

    word_w2v, w2v_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(word_vob)))
    print("source_W  size: " + str(len(word_W)))
    print("num words in source word2vec: " + str(len(word_w2v)))

    # type_k, type_W = load_vec_random(TYPE_vob, k=w2v_k)
    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=t2v_k)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))


    max_posi = 20
    posi_k, posi_W = load_vec_onehot(k=max_posi + 1)
    print('posi_k, posi_W', posi_k, len(posi_W))


    # weigtnum = int(len(fragment_train) * percent)
    # fragment_train = fragment_train[:weigtnum]

    pairs_train, labels_train, classifer_labels_train = \
        CreatePairs(trainfile, max_s, max_posi, word_vob, target_vob, type_W, char_vob, max_c, istest=False)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))

    pairs_test, labels_test, classifer_labels_test = \
        CreatePairs(testfile, max_s, max_posi, word_vob, target_vob, type_W, char_vob, max_c, istest=True)
    print('CreatePairs test len = ', len(pairs_test[0]), len(labels_test))


    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([pairs_train, labels_train, classifer_labels_train,
                 pairs_test, labels_test, classifer_labels_test,
                word_vob, word_id2word, word_W, w2v_k,
                 char_vob, char_id2char, char_W, c2v_k,
                 target_vob, target_id2word, type_W, type_k,
                 posi_W, posi_k,
                max_s, max_posi, max_c], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    alpha = 10
    maxlen = 50
    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    t2v_file = './data/KG2v/FB15K_OpenKETransE_Relation2Vec_100.txt'
    trainfile = './data/annotated_fb__zeroshot_RE.random.train.txt'
    testfile = './data/annotated_fb__zeroshot_RE.random.test.txt'
    resultdir = "./data/result/"

    word_vob, word_id2word, target_vob, target_id2word, max_s = get_word_index({trainfile, testfile})
    print("source vocab size: ", str(len(word_vob)))
    print("word_id2word size: ", str(len(word_id2word)))

    # if max_s > maxlen:
    #     max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))



    # type_k, type_W = load_vec_random(TYPE_vob, k=w2v_k)
    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=100)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))


    max_posi = 20
    posi_k, posi_W = load_vec_onehot(k=max_posi + 1)
    print('posi_k, posi_W', posi_k, len(posi_W))


    # weigtnum = int(len(fragment_train) * percent)
    # fragment_train = fragment_train[:weigtnum]

    pairs_train, labels_train, classifer_labels_train = \
        CreatePairs(trainfile, max_s, max_posi, target_vob, type_W, word_id2word, char_vob=None, max_c=-1, istest=False)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))


