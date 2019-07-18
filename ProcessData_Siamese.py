#coding=utf-8
__author__ = 'JIA'
import numpy as np
import pickle, codecs
import json
import re, random
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


def make_idx_character_index_withFix(file, max_s, max_c, source_vob):

    data_s_all = []
    count = 0
    f = open(file,'r')
    fr = f.readlines()

    # prefixs = []
    # prefix = open("./data/EnFix/EnPrefix.txt", 'r')
    # pf = prefix.readlines()
    # for line in pf:
    #     prefixs.append(line)

    suffixs = []
    suffix = open("./data/EnFix/EnSuffix.txt", 'r')
    sf = suffix.readlines()
    for line in sf:
        suffixs.append(line)

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
            count = 0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]
        count_c = 0
        start = 0
        end = 0
        # print(word)
        # for pre in prefixs:
        #     pre = pre.strip('\r\n').rstrip('\n').rstrip('\r')
        #     if re.match(pre, word, flags=re.I) is not None:
        #         character = pre
        #         data_c.append(source_vob[character])
        #         start = character.__len__()
        #         count_c +=1
        #         break

        endidex = 0
        for suf in suffixs:
            suf = suf.strip('\r\n').rstrip('\n').rstrip('\r')
            if re.search(suf + '$', word, flags=re.I) is not None:

                character = suf
                endidex=source_vob[character]
                end = character.__len__()

                break

        for chr in range(start, min(word.__len__() - end, max_c)):
            count_c += 1
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])
        if count_c < max_c:
            data_c.append(endidex)
            count_c += 1

        num = max_c - count_c
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count +=1
        # print(data_c.__len__())
        # print(data_c)
        if data_c.__len__() != max_c:
            while 1>0:
                i=1
        data_w.append(data_c)

    f.close()
    return data_s_all


def make_idx_POS_index(file, max_s, source_vob, Poswidth=3):

    width = (Poswidth-1)//2

    count = 0
    data_s_all = []
    data_s = []

    f = open(file,'r')
    fr = f.readlines()
    sen_i = 0

    for i, line in enumerate(fr):

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append([0] * Poswidth)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_s = []
            count = 0
            sen_i = 0
            continue

        data_w = []


        for k in range(width, 0, -1):
            if sen_i - k < 0:
                data_w.append(0)
                # print('>>>')
            else:
                sourc_pre = fr[i - k].strip('\r\n').rstrip('\n').split(' ')[1]
                data_w.append(source_vob[sourc_pre])
                # print(sourc_pre)

        sent = line.strip('\r\n').rstrip('\n').split(' ')[1]
        if not source_vob.__contains__(sent):
            data_w.append(source_vob["**UNK**"])
        else:
            data_w.append(source_vob[sent])
        # print(sent)

        for k in range(1, width+1):
            if i + k >= fr.__len__() or fr[i + k].__len__() <= 1:
                for s in range(k, width+1):
                    data_w.append(0)
                    # print('<<<')
                break
            else:
                sourc_back = fr[i + k].strip('\r\n').rstrip('\n').split(' ')[1]
                data_w.append(source_vob[sourc_back])
                # print(sourc_back)

        data_s.append(data_w)
        if len(data_w) is not Poswidth:
            print('____________________', data_w)
        count += 1
        sen_i += 1

    f.close()
    # print(data_t_all)
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


def get_Feature_index(file):
    """
    Give each feature labelling an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0

    for labelingfile in file:
        f = open(labelingfile, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:

                continue

            sourc = line.strip('\r\n').rstrip('\n').split(' ')[1]
            # print(sourc)
            if not label_vob.__contains__(sourc):
                label_vob[sourc] = count
                label_idex_word[count] = sourc
                count += 1

        f.close()
    if not label_vob.__contains__("**UNK**"):
        label_vob["**UNK**"] = count
        label_idex_word[count] = "**UNK**"
        count += 1


    return label_vob, label_idex_word


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
        f = open(file, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue

            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')[0]
            # if sourc.__len__() > max_c:
            #     max_c = sourc.__len__()
            #     print(sourc)

            for character in sourc:
                if not source_vob.__contains__(character):
                    source_vob[character] = count
                    sourc_idex_word[count] = character
                    count += 1

        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_c


def get_Character_index_withFix(files):


    source_vob = {}
    sourc_idex_word = {}
    max_c = 18-5
    count = 1

    # prefixs =[]
    # prefix = open("./data/EnFix/EnPrefix.txt", 'r')
    # pf = prefix.readlines()
    # for line in pf:
    #     prefixs.append(line)

    suffixs = []
    suffix = open("./data/EnFix/EnSuffix.txt", 'r')
    sf = suffix.readlines()
    for line in sf:
        suffixs.append(line)

    for file in files:
        f = open(file, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue

            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')[0]
            # if sourc.__len__() > max_c:
            #     max_c = sourc.__len__()
            #     print(sourc)
            start = 0
            end = 0
            # for pre in prefixs:
            #     pre = pre.strip('\r\n').rstrip('\n').rstrip('\r')
            #     if re.match(pre, sourc, flags=re.I) is not None:
            #         # print('1@@@@@@@@@@@@@', pre)
            #         character = pre
            #         if not source_vob.__contains__(character):
            #             source_vob[character] = count
            #             sourc_idex_word[count] = character
            #             count += 1
            #             start = character.__len__()
            #         break

            for suf in suffixs:
                suf = suf.strip('\r\n').rstrip('\n').rstrip('\r')
                if re.search(suf + '$', sourc, flags=re.I) is not None:
                    # print('2#############', suf)
                    character = suf
                    if not source_vob.__contains__(character):
                        source_vob[character] = count
                        sourc_idex_word[count] = character
                        count += 1
                        end = character.__len__()
                    break
            # t = sourc.__len__()
            # if end is not 0:
            #     t = sourc.__len__() - end + 1
            # if t > max_c:
            #     max_c = t
            #     print('max_c', max_c, sourc)

            for i in range(start, sourc.__len__()-end):
                character = sourc[i]
                if not source_vob.__contains__(character):
                    source_vob[character] = count
                    sourc_idex_word[count] = character
                    count += 1

        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_c


def ReadfromTXT2Lists(file, source_vob, target_vob):

    ner_count = 0

    sen2list = []
    tag2list = []

    sen2list_all = []
    tag2list_all = []

    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:

        if line.__len__() <= 1:

            sen2list_all.append(sen2list)
            tag2list_all.append(tag2list)

            sen2list = []
            tag2list = []

            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')

        if not source_vob.__contains__(sent[0]):
            sen2list.append(source_vob["**UNK**"])
        else:
            sen2list.append(source_vob[sent[0]])

        tag2list.append(target_vob[sent[4]])

    f.close()

    return sen2list_all, tag2list_all


def Lists2Set(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment, hasNeg=False):
    fragment_list = []

    assert len(sen2list_all) == len(tag2list_all)
    for id, tag2list in enumerate(tag2list_all):

        target_left = 0
        fragtuples_list = []
        for index, tag in enumerate(tag2list):

            if target_idex_word[tag] == 'O':
                target_left = index
                continue

            else:
                if target_idex_word[tag].__contains__('B-'):
                    target_left = index

                elif target_idex_word[tag].__contains__('I-'):
                    continue

                elif target_idex_word[tag].__contains__('S-'):

                    target_left = index
                    target_right = index+1
                    reltag = target_idex_word[tag][2:]
                    tuple = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple)
                    if hasNeg:
                        inc = random.randrange(0, 5)
                        if inc == 0:
                            inc = random.randrange(-2, 3)
                            reltag = 'NULL'

                            if inc == 0:
                                target_left = max(0, target_left-1)
                                target_right = min(len(tag2list), target_right+1)
                            elif inc < 0:
                                target_left = max(0, target_left + inc)
                            else:
                                target_right = min(len(tag2list), target_right + inc)

                            tuple = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                            fragtuples_list.append(tuple)


                    flens = max(index+1, len(tag2list)-index)
                    if flens > max_context:
                        max_context = flens

                    target_left = index

                elif target_idex_word[tag].__contains__('E-'):
                    target_right = index+1
                    reltag = target_idex_word[tag][2:]
                    tuple = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple)

                    flens = max(index+1, len(tag2list)-target_left)
                    if flens > max_context:
                        max_context = flens

                    max_fragment = max(max_fragment, target_right-target_left)

                    if hasNeg:
                        inc = random.randrange(0, 5)
                        if inc == 0:
                            inc = random.randrange(-2, 3)
                            reltag = 'NULL'

                            if inc == 0:
                                target_left = max(0, target_left - 1)
                                target_right = min(len(tag2list), target_right + 1)
                            elif inc < 0:
                                target_left = max(0, target_left + inc)
                            else:
                                target_right = min(len(tag2list), target_right + inc)

                            tuple = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                            fragtuples_list.append(tuple)
                            max_fragment = max(max_fragment, target_right - target_left)

                    target_left = index

                else:
                    print("Seq2frag error !!!!!!!!")

        for tup in fragtuples_list:
            context_left = tup[1]
            fragment_left = tup[2]
            fragment_right = tup[3]
            context_right = tup[4]
            fragment_tag = tup[6]
            fragment_list.append((fragment_left, fragment_right, fragment_tag, sen2list_all[id], context_left, context_right))

    return fragment_list, max_context, max_fragment


def CreatePairs(trainfile, max_s, max_posi, target_vob, type_W, word_id2word, char_vob, max_c):

    labels = []
    data_s_all = []
    data_e1_posi_all = []
    data_e2_posi_all = []
    data_tag_all = []
    classifer_label = []


    f = codecs.open(trainfile, 'r', encoding='utf-8')
    for line in f.readlines():
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        sent = jline['sent'].split(' ')
        rel = jline['rel']
        e1_l = jline['e1_l']
        e1_r = jline['e1_r']
        e2_l = jline['e2_l']
        e2_r = jline['e2_r']

        data_tag = target_vob[rel]

        data_s = sent[0:min(len(sent), max_s)] + [0] * max(0, max_s - len(sent))

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



        data_s_all.append(data_s)
        data_tag_all.append(type_W[data_tag])
        data_e1_posi_all.append(data_e1_posi)
        data_e2_posi_all.append(data_e2_posi)
        labels.append(1)
        classifer_label.append(data_tag)

        inc = random.randrange(1, len(target_vob.keys()))
        dn = (data_tag + inc) % len(target_vob.keys())
        data_s_all.append(data_s)
        data_tag_all.append(type_W[dn])
        data_e1_posi_all.append(data_e1_posi)
        data_e2_posi_all.append(data_e2_posi)
        labels.append(0)
        classifer_label.append(data_tag)

    pairs = [data_s_all, data_tag_all,
             data_e1_posi_all, data_e2_posi_all]

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
    print("target vocab size: " + str(target_vob))
    print("target_id2word size: " + str(target_id2word))
    # if max_s > maxlen:
    #     max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))

    char_vob = {}
    char_id2char = {}
    max_c = -1
    # char_vob, char_id2char, max_c = get_Character_index({trainfile, devfile, testfile})
    # print("source char size: ", char_vob.__len__())
    # print("max_c: ", max_c)
    # print("source char: " + str(char_id2char))
    char_W = {}
    # char_W, char_k = load_vec_character(c2v_file, char_vob, c2v_k)
    # print('character_W shape:', char_W.shape)

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

    pairs_train, labels_train, classifer_labels_train = CreatePairs(trainfile, max_s, max_posi, target_vob, type_W, word_id2word, char_vob=None, max_c=-1)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))


    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([pairs_train, labels_train, classifer_labels_train,
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
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    datafile = "./data/model/data.pkl"
    modelfile = "./data/model/model.pkl"
    resultdir = "./data/result/"

