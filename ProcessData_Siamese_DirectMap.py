#coding=utf-8

__author__ = 'JIA'
import numpy as np
import pickle, codecs
import json
import re, random, math
import keras
from keras.utils.np_utils import to_categorical


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
            unknowtoken += 1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[lower_word]

    print('UnKnown tokens in w2v', unknowtoken)
    return w2v, k, W


def load_vec_KGrepresentation(fname, vocab, k):

    f = codecs.open(fname, 'r', encoding='utf-8')
    w2v = {}
    for line in f.readlines():

        values = line.rstrip('\n').split()
        word = ' '.join(values[:len(values)-100])
        coefs = np.asarray(values[len(values)-100:], dtype='float32')
        w2v[word] = coefs
    f.close()

    W = np.zeros(shape=(vocab.__len__(), k))
    for item in vocab:

        try:
            W[vocab[item]] = w2v[item]
        except BaseException:
            print('the rel is not finded ...', item)

    return k, W


def get_relembed_sim_rank(type_W):

    k = len(type_W)
    W = np.zeros(shape=(k, k))
    RankDict = {}
    for i in range(0, len(type_W)):

        i_j = []
        for j in range(0, len(type_W)):

            vector_a = np.mat(type_W[i])
            vector_b = np.mat(type_W[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            i_j.append(cos)

        try:
            coefs = np.asarray(i_j, dtype='float32')
            W[i] = coefs
        except BaseException:
            print('the rel is not finded ...', i)

        # ijlist = sorted(i_j.items(), key=lambda x: x[1], reverse=True)

        # print('------------', target_id2word[i])
        # for si, s in enumerate(ijlist):
        #     if si > 10:
        #         break
        #     print(target_id2word[s[0]], s[1])

        # ijlist = dict(ijlist)
        # ijlist = list(ijlist.keys())
        # RankDict[i] = ijlist

        # RankDict[i] = list(i_j.values())

        # rank = {}
        # for p, tup in enumerate(ijlist):
        #     rank[tup[0]] = p+1
        # RankDict[i] = rank

        # print(RankDict[i])

    return k, W


def load_vec_ClassSimRank(fname, vocab, k=120):

    W = np.zeros(shape=(k, k))

    f = codecs.open(fname, 'r', encoding='utf-8')

    for line in f.readlines():

        values = line.rstrip('\n').split()
        word = ' '.join(values[:len(values)-k])
        coefs = np.asarray(values[len(values)-k:], dtype='float32')
        try:
            W[vocab[word]] = coefs
        except BaseException:
            print('the rel is not finded ...', line)

    f.close()

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
    target_vob_train = {}
    count = 1
    tarcount = 0

    max_s = 0
    max_l = 0

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
            rel_words = rel.split(' ')
            for word in rel_words:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1

            if not target_vob.__contains__(rel):
                target_vob[rel] = tarcount
                target_idex_word[tarcount] = rel
                tarcount += 1

            if testf == files[0]:
                if not target_vob_train.__contains__(rel):
                    target_vob_train[rel] = target_vob[rel]

            max_s = max(max_s, len(words))
            max_l = max(max_l, len(rel_words))

        f.close()

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, target_vob_train, max_l


def get_Character_index(files):


    source_vob = {}
    sourc_idex_word = {}
    max_c = 0
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
                max_c = max(max_c, len(word))
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


def get_prototypes_byques(target_vob, word_vob):

    quesfile = './data/WikiReading/WikiReading.quenstion.txt'

    tagDict = {}

    f = codecs.open(quesfile, 'r', encoding='utf-8')
    lines = f.readlines()
    max_s = 245
    for si, line in enumerate(lines):
        jline = line.rstrip('\n').split(' \t ')
        sent = jline[1].split(' ')
        rel = target_vob[jline[0]]

        if rel not in tagDict:
            tagDict[rel] = []

        data_s = []
        for ww in sent[0:min(len(sent), max_s)]:
            if ww not in word_vob:
                # word_vob[ww] = word_vob['**UNK**']
                data_s.append(word_vob['**UNK**'])
            else:
                data_s.append(word_vob[ww])
        data_s = data_s + [0] * max(0, max_s - len(sent))


        tagDict[rel] = data_s

    print(len(tagDict))

    f.close()

    return max_s, tagDict



def get_sentDicts_neg(trainfile, max_s, max_posi, word_vob, char_vob, max_c):

    tagDict = {}

    tagDict[-1] = []

    f = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f.readlines()

    for si, line in enumerate(lines):
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_posi'][0]
        e1_r = jline['e1_posi'][1]
        e2_l = jline['e2_posi'][0]
        e2_r = jline['e2_posi'][1]

        max_long = max(e1_r, e2_r)
        if len(sent) > max_s and max_long > max_s:
            continue

        data_tag = -1

        # word_vob['____'] = len(word_vob)+1

        data_s = []
        for ww in sent[0:min(len(sent), max_s)]:
            if ww not in word_vob:
                data_s.append(word_vob['**UNK**'])
            else:
                data_s.append(word_vob[ww])
        data_s = data_s + [0] * max(0, max_s - len(sent))

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
        for wi in range(0, min(len(sent), max_s)):
            word = sent[wi]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])
            data_c = data_c + [0] * max(max_c - word.__len__(), 0)
            char_s.append(data_c)
        char_s = char_s + [[0] * max_c] * max(0, max_s - len(char_s))

        pairs = [data_s, data_e1_posi, data_e2_posi, char_s]


        if data_tag not in tagDict.keys():
            tagDict[data_tag] = []
            # if prototypes != None and data_tag in prototypes.keys():
            #     tagDict[data_tag].append(prototypes[data_tag][0])

        tagDict[data_tag].append(pairs)

    f.close()

    return tagDict


def get_sentDicts(trainfile, max_s, max_posi, word_vob, target_vob, char_vob, max_c,
                  istest=False, needDEV=False, target_vob_4dev=None, prototypes=None):

    tagDict = {}
    tagDict_dev = {}

    f = codecs.open(trainfile, 'r', encoding='utf-8')
    lines = f.readlines()

    thd = -1
    if needDEV == True:
        # thd = len(lines) * 0.15
        thd = len(lines) * 0.08

    for si, line in enumerate(lines):
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_posi'][0]
        e1_r = jline['e1_posi'][1]
        e2_l = jline['e2_posi'][0]
        e2_r = jline['e2_posi'][1]

        max_long = max(e1_r, e2_r)
        if len(sent) > max_s and max_long > max_s:
            continue

        data_tag = target_vob[rel]

        # word_vob['____'] = len(word_vob)+1

        data_s = []
        for ww in sent[0:min(len(sent), max_s)]:
            if ww not in word_vob:
                word_vob[ww] = word_vob['**UNK**']
            data_s.append(word_vob[ww])
        data_s = data_s + [0] * max(0, max_s - len(sent))

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
        for wi in range(0, min(len(sent), max_s)):
            word = sent[wi]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])
            data_c = data_c + [0] * max(max_c - word.__len__(), 0)
            char_s.append(data_c)
        char_s = char_s + [[0] * max_c] * max(0, max_s - len(char_s))

        pairs = [data_s, data_e1_posi, data_e2_posi, char_s]

        # if needDEV == True and si < thd:
        # if needDEV == True and rel in target_vob_4dev.keys():
        if needDEV == True and (si < thd or rel in target_vob_4dev.keys()):

            if data_tag not in tagDict_dev.keys():

                tagDict_dev[data_tag] = []

                # if prototypes != None and data_tag in prototypes.keys():
                #     tagDict_dev[data_tag].append(prototypes[data_tag][0])

            tagDict_dev[data_tag].append(pairs)


        # elif istest == True:
        #
        #     if data_tag not in tagDict.keys():
        #         if prototypes != None:
        #             tagDict[data_tag] = prototypes[data_tag]
        #         else:
        #             tagDict[data_tag] = []
        #     if len(tagDict[data_tag]) < 400:
        #         tagDict[data_tag].append(pairs)

        else:

            if data_tag not in tagDict.keys():
                tagDict[data_tag] = []
                # if prototypes != None and data_tag in prototypes.keys():
                #     tagDict[data_tag].append(prototypes[data_tag][0])

            tagDict[data_tag].append(pairs)

    f.close()

    return tagDict, tagDict_dev



def get_classDicts(max_l, source_vob, target_vob):

    classDict = {}

    for si, line in enumerate(target_vob.keys()):

        rel = line
        sent = rel.split(' ')

        data_s = []
        for ww in sent[0:min(len(sent), max_l)]:
            # if ww not in word_vob:
            #     word_vob[ww] = word_vob['**UNK**']
            data_s.append(source_vob[ww])
        data_s = data_s + [0] * max(0, max_l - len(sent))

        classDict[target_vob[rel]] = data_s


    return classDict


def CreatePairs(tagDict_train, istest=False):

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            labels.append(0)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_tag_all]

    return pairs, labels


def CreateTriplet(tagDict_train, target_vob=None, istest=False):

    categorical_labels = None
    if target_vob != None:
        categorical_labels = to_categorical(list(target_vob.values()), num_classes=None)
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1
            if target_vob != None:
                labels.append(categorical_labels[tag])
            else:
                labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_sample(tagDict_train, target_vob=None, sample_n=2000):

    categorical_labels = None
    if target_vob != None:
        categorical_labels = to_categorical(list(target_vob.values()), num_classes=None)
    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc0 = random.randrange(0, len(sents))
        inc1 = random.randrange(0, len(sents))
        if inc1 == inc0:
            inc1 = (inc1 + 1) % len(sents)
        i = 0
        count = 0
        while i < len(sents):
            if count >= sample_n:
                break

            p0 = (inc0 + i) % len(sents)
            p1 = (inc1 + i) % len(sents)

            i += 1
            if target_vob != None:
                labels.append(categorical_labels[tag])
            else:
                labels.append(1)
            data_tag_all.append([tag])
            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            count += 1
            if i == len(sents):
                i = 0
                inc0 = (inc0 + random.randrange(1, len(sents))) % len(sents)
                inc1 = (inc1 + random.randrange(1, len(sents))) % len(sents)
                if inc1 == inc0:
                    inc1 = (inc1 + 1) % len(sents)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_DirectMAP(tagDict_train, classNameDict, target_vob=None, istest=False):

    labels = []
    data_tag_all_p = []
    data_tag_all_n = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1

            labels.append(1)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_tag_all_p.append(classNameDict[tag])

            if target_vob != None:
                keylist = list(target_vob.values())
            else:
                keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)

            data_tag_all_n.append(classNameDict[keylist[ran1]])

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_tag_all_p, data_tag_all_n]

    return pairs, labels


def CreateTriplet_withSoftmax(tagDict_train, target_vob=None, istest=False):

    """
    if target_vob != None, unseen classes are used for pos examples
    """

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1


            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = np.zeros(2)
            if i % 2 == 0:
                targetvec[0] = 1

                if target_vob == None:
                    data_tag_all.append([keylist[ran1]])
                else:
                    keylist = list(target_vob.values())
                    ran1 = random.randrange(0, len(keylist))
                    if keylist[ran1] == tag:
                        ran1 = (ran1 + 1) % len(keylist)
                    data_tag_all.append([keylist[ran1]])
            else:
                targetvec[1] = 1
                data_tag_all.append([tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def CreateTriplet_withSoftmax_ques(tagDict_train, prototype_tagDict, target_vob=None, istest=False):

    """
    if target_vob != None, unseen classes are used for pos examples
    """

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1


            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = np.zeros(2)
            if i % 2 == 0:
                targetvec[0] = 1

                if target_vob == None:
                    data_tag_all.append(prototype_tagDict[keylist[ran1]])
                else:
                    keylist = list(target_vob.values())
                    ran1 = random.randrange(0, len(keylist))
                    if keylist[ran1] == tag:
                        ran1 = (ran1 + 1) % len(keylist)
                    data_tag_all.append(prototype_tagDict[keylist[ran1]])
            else:
                targetvec[1] = 1
                data_tag_all.append(prototype_tagDict[tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels



def CreateTriplet_withMSE(tagDict_train):

    labels = []
    data_tag_all = []

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_s_all_2 = []
    data_e1_posi_all_2 = []
    data_e2_posi_all_2 = []
    char_s_all_2 = []


    for tag in tagDict_train.keys():
        sents = tagDict_train[tag]

        if len(sents) < 2:
            continue
        inc = random.randrange(1, len(sents))
        i = 0
        while i < len(sents):
            p0 = i
            p1 = (inc + i) % len(sents)

            i += 1


            data_s, data_e1_posi, data_e2_posi, char_s = sents[p0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_s, data_e1_posi, data_e2_posi, char_s = sents[p1]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)

            keylist = list(tagDict_train.keys())
            ran1 = random.randrange(0, len(keylist))
            if keylist[ran1] == tag:
                ran1 = (ran1 + 1) % len(keylist)
            ran2 = random.randrange(0, len(tagDict_train[keylist[ran1]]))
            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_train[keylist[ran1]][ran2]
            data_s_all_2.append(data_s)
            data_e1_posi_all_2.append(data_e1_posi)
            data_e2_posi_all_2.append(data_e2_posi)
            char_s_all_2.append(char_s)

            targetvec = 0
            if i % 2 == 0:
                targetvec = 0
                data_tag_all.append([keylist[ran1]])

            else:
                targetvec = 1
                data_tag_all.append([tag])
            labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1,
             data_s_all_2, data_e1_posi_all_2, data_e2_posi_all_2, char_s_all_2,
             data_tag_all]

    return pairs, labels


def get_split_train_dev(target_id2word):

    rel4dev = {}

    for i in [7, 14, 25, 39, 43, 55, 60, 72, 83]:
        rel4dev[target_id2word[i]] = i
    return rel4dev


def get_rel_prototypes(file, max_s, max_posi, word_vob, target_vob, char_vob, max_c):

    tagDict_prototypes, _ = get_sentDicts(file, max_s, max_posi, word_vob, target_vob, char_vob, max_c, needDEV=False)
    print('tagDict_prototypes len', len(tagDict_prototypes))

    return tagDict_prototypes


def get_data(trainfile, testfile, prototypesfile, w2v_file, c2v_file, t2v_file, datafile, w2v_k=300, c2v_k=25, t2v_k=100, maxlen = 50,
             hasNeg=False, percent=1):

    """
    数据处理的入口函数
    Converts the input files  into the model input formats
    """

    word_vob, word_id2word, target_vob, target_id2word, max_s, target_vob_train, max_l = get_word_index([trainfile, testfile, prototypesfile])
    print("source vocab size: ", str(len(word_vob)))
    print("word_id2word size: ", str(len(word_id2word)))
    print("target vocab size: " + str(len(target_vob)))
    print("target_id2word size: " + str(len(target_id2word)))
    if max_s > maxlen:
        max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))
    print('max_l', max_l)


    char_vob, char_id2char, max_c = get_Character_index({trainfile, testfile, prototypesfile})
    print("source char size: ", char_vob.__len__())
    max_c = min(max_c, 18)
    print("max_c: ", max_c)

    c2v_k, char_W, = load_vec_random(char_vob, k=c2v_k)
    print('character_W shape:', char_W.shape)

    word_w2v, w2v_k, word_W = load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(word_vob)))
    print("source_W  size: " + str(len(word_W)))


    type_k, type_W = load_vec_KGrepresentation(t2v_file, target_vob, k=t2v_k)
    print('TYPE_k, TYPE_W', type_k, len(type_W[0]))

    classNameDict = get_classDicts(max_l, word_vob, target_vob)




    max_posi = 20
    posi_k, posi_W = load_vec_onehot(k=max_posi + 1)
    print('posi_k, posi_W', posi_k, len(posi_W))


    # weigtnum = int(len(fragment_train) * percent)
    # fragment_train = fragment_train[:weigtnum]

    tagDict_prototypes, _ = get_sentDicts(prototypesfile, max_s, max_posi, word_vob, target_vob, char_vob, max_c)
    print('tagDict_prototypes len', len(tagDict_prototypes))

    tagDict_test, tagDict_dev = get_sentDicts(testfile, max_s, max_posi, word_vob, target_vob, char_vob, max_c,
                                              istest=False, prototypes=tagDict_prototypes)
    assert tagDict_dev == {}
    print('tagDict_test len', len(tagDict_test))

    target_vob_4dev = get_split_train_dev(target_id2word)
    print('target_vob len', len(target_vob), 'target_vob_4dev len', len(target_vob_4dev))

    tagDict_train, tagDict_dev = get_sentDicts(trainfile, max_s, max_posi, word_vob, target_vob, char_vob, max_c,
                                               needDEV=True, target_vob_4dev=target_vob_4dev, prototypes=tagDict_prototypes)
    print('tagDict_train len', len(tagDict_train), 'tagDict_dev len', len(tagDict_dev))

    # pairs_train, labels_train = CreatePairs(tagDict_train, istest=False)
    # print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))
    #
    # pairs_test, labels_test = CreatePairs(tagDict_test, istest=True)
    # print('CreatePairs test len = ', len(pairs_test[0]), len(labels_test))

    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([tagDict_train, tagDict_dev, tagDict_test,
                word_vob, word_id2word, word_W, w2v_k,
                 char_vob, char_id2char, char_W, c2v_k,
                 target_vob, target_id2word,
                 posi_W, posi_k, type_W, type_k,
                 classNameDict,
                max_s, max_posi, max_c, max_l], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    alpha = 10
    maxlen = 50

    trainfile = './data/WikiReading/WikiReading_data.random.train.txt'
    testfile = './data/WikiReading/WikiReading_data.random.test.txt'
    rel_prototypes_file = './data/WikiReading/rel_class_prototypes.txt.json.txt'

    word_vob, word_id2word, target_vob, target_id2word, max_s, target_vob_train, max_l = get_word_index([trainfile, testfile, rel_prototypes_file])
    print("source vocab size: ", str(len(word_vob)))
    print("word_id2word size: ", str(len(word_id2word)))
    print("target vocab size: " + str(len(target_vob)))
    print("target_id2word size: " + str(len(target_id2word)))
    if max_s > maxlen:
        max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))
    print('max_l', max_l)

