
import tensorflow as k
import codecs, json
import numpy as np


def Process_Corpus():
    f = '/Users/shengbinjia/Downloads/FewRel-master/data/val.json'
    fw1w = codecs.open(f + '.FewRel_data.test.txt', 'w', encoding='utf-8')

    frr2 = codecs.open(f, 'r', encoding='utf-8')
    jdict = {}
    for line in frr2.readlines():

        sent = json.loads(line.strip('\r\n').strip('\n'))
        jdict = dict(sent)


    # jlist = sorted(list(jdict.keys()))
    dict4w = {}
    for ki, k in enumerate(jdict.keys()):
        jlist = jdict[k]

        i = 0
        for sublist in jlist:

            i += 1
            tokens = sublist['tokens']
            e1 = sublist['h']
            e2 = sublist['t']
            # print(ki, k, i, tokens)
            print(e1, e2)
            sent = ''
            for word in tokens:
                sent += ' ' + word
            dict4w['sent'] = sent[1:]
            dict4w['rel'] = k

            dict4w['e1_name'] = e1[0]
            dict4w['e1_id'] = e1[1]
            e1_posi = [e1[2][0][0], e1[2][0][0] + len(e1[2][0])]
            dict4w['e1_posi'] = e1_posi

            dict4w['e2_name'] = e2[0]
            dict4w['e2_id'] = e2[1]
            e2_posi = [e2[2][0][0], e2[2][0][0] + len(e2[2][0])]
            dict4w['e2_posi'] = e2_posi

            fj = json.dumps(dict4w, ensure_ascii=False)
            fw1w.write(fj + '\n')
        # print(i)
    frr2.close()
    fw1w.close()


def load_vec_txt(fname, vocab, k=300):
    w2v_file = "./data/w2v/glove.6B.100d.txt"

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


def get_rel2v_ave_glove100():

    id2name = {}
    f = './data/FewRel/FewRel_rel_id_name.txt'
    fr = codecs.open(f, 'r', encoding='utf-8')
    lines = fr.readlines()
    for line in lines:
        lsp = line.rstrip('\n').split('\t')
        id = lsp[0]
        name = lsp[1]
        id2name[id] = name
    fr.close()
    print(len(id2name))

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    fr = codecs.open(w2v_file, 'r', encoding='utf-8')
    w2v = {}
    unknowtoken = 0
    for line in fr.readlines():
        values = line.rstrip('\n').split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    fr.close()
    print(len(w2v))

    # i = j = 0
    #
    # for ki in id2name.keys():
    #     for w in id2name[ki].split():
    #         print(w)
    #         i += 1
    #         if w not in w2v.keys():
    #             j += 1
    #             print(w)
    # print(i, j)

    rel2v_file = "./data/FewRel/FewRel.rel2v.by_glove.100d.txt"
    fw = open(rel2v_file, 'w', encoding='utf-8')

    for ki in id2name.keys():
        print(ki)
        words = id2name[ki].split()
        W = np.zeros(shape=(words.__len__(), 100))
        for wi, w in enumerate(words):
            lower_word = w.lower()
            print(w2v[lower_word])
            W[wi] = w2v[lower_word]
        ave = np.mean(W, axis=0)
        string = ki
        for item in ave:
            string += ' ' + str(item)
        fw.write(string + '\n')
    fw.close()



if __name__ == '__main__':

    # Process_Corpus()

    get_rel2v_ave_glove100()