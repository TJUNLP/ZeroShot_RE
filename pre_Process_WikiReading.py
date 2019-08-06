
import tensorflow as k
import codecs, json, random, re, nltk
import numpy as np


def Process_Corpus():

    f = './data/WikiReading/WikiReading.txt'
    fw1w = codecs.open(f + '.json.txt', 'w', encoding='utf-8')
    frr = codecs.open(f, 'r', encoding='utf-8')
    jsondict = {}
    cc = 0
    for line in frr.readlines():
        ls = line.rstrip('\n').split('\t')

        sent0 = ls[3]

        en1_name0 = ls[2]
        en1_name1 = '_en1_name1_'
        sent1 = sent0.replace(en1_name0, en1_name1)
        en2_name0 = ls[4]
        en2_name1 = '_en2_name1_'
        sent2 = sent1.replace(en2_name0, en2_name1)

        if '_en1_name1_' not in sent2 or '_en2_name1_' not in sent2:
            # print('----------\n')
            # print(line)
            # print(en1_name1, en1_name0)
            # print(en2_name1, en2_name0)
            # print(sent0)
            # print(sent2)
            # cc += 1
            # print(cc)
            continue


        sent3 = ' '.join(nltk.word_tokenize(sent2))

        sent4 = sent3.replace(en1_name1, en1_name0)
        sent5 = sent4.replace(en2_name1, en2_name0)

        try:
            assert en1_name0 in sent5
            assert en2_name0 in sent5
        except:
            print('----------\n')
            # print(line)
            # print(en1_name1, en1_name0)
            # print(en2_name1, en2_name0)
            # print(sent0)
            # print(sent3)
            # print(sent5)
            continue

        jsondict['sent'] = sent5
        jsondict['en1_name'] = ls[2]
        jsondict['en2_name'] = ls[4]
        jsondict['rel'] = ls[0]


        en1_name = re.sub(r'\(', '\\\(', en1_name0)
        en1_name = re.sub(r'\)', '\\\)', en1_name)
        en2_name = re.sub(r'\(', '\\\(', en2_name0)
        en2_name = re.sub(r'\)', '\\\)', en2_name)
        en1_name = re.sub(r'\?', '\\\?', en1_name)
        en2_name = re.sub(r'\?', '\\\?', en2_name)
        en1_name = re.sub(r'\+', '\\\+', en1_name)
        en2_name = re.sub(r'\+', '\\\+', en2_name)
        en1_name = re.sub(r'\$', '\\\$', en1_name)
        en2_name = re.sub(r'\$', '\\\$', en2_name)
        en1_name = re.sub(r'\*', '\\\*', en1_name)
        en2_name = re.sub(r'\*', '\\\*', en2_name)

        e1_0, e1_1 = re.search(en1_name, sent5, flags=re.I).span()
        e1_l = len(re.compile(r' ').findall(sent5[:e1_0]))
        e1_r = len(re.compile(r' ').findall(sent5[:e1_1])) + 1

        tmplist = list(sent5)
        tmplist2 = tmplist[:e1_0] + ['_' for x in range(e1_0, e1_1)] + tmplist[e1_1:]
        sent6 = ''.join(tmplist2)
        # print(line)
        e2_0, e2_1 = re.search(en2_name, sent6, flags=re.I).span()
        e2_l = len(re.compile(r' ').findall(sent5[:e2_0]))
        e2_r = len(re.compile(r' ').findall(sent5[:e2_1])) + 1
        # print(e1_l, e1_r, e2_l, e2_r)

        if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
            print('>>>>>>>>>>>>>\n', sent5)
            print(line)
            print(e1_l, e1_r, e2_l, e2_r)
        else:
            if e1_r <= e2_l or e2_r <= e1_l:
                jsondict['e1_posi'] = (e1_l, e1_r)
                jsondict['e2_posi'] = (e2_l, e2_r)
            else:
                print('..............\n', sent5)
                print(line)
                print(e1_l, e1_r, e2_l, e2_r)

        fj = json.dumps(jsondict, ensure_ascii=False)
        fw1w.write(fj + '\n')


    frr.close()
    fw1w.close()

    # frr2 = codecs.open(f + '.2.txt', 'r', encoding='utf-8')
    # for line in frr2.readlines():
    #
    #     sent = json.loads(line.strip('\r\n').strip('\n'))
    #     print(sent['sent'])
    #     poslist = sent['pos']
    #     for pos in poslist:
    #         print(pos[1])


def Split_zeroshotData_2_train_test():

    # fw_train = './data/WikiReading/WikiReading_data.random.train.txt'
    # fw_test = './data/WikiReading/WikiReading_data.random.test.txt'
    # fwtr = codecs.open(fw_train, 'w', encoding='utf-8')
    # fwte = codecs.open(fw_test, 'w', encoding='utf-8')

    fr = './data/WikiReading/WikiReading.txt.json.txt'
    frr = codecs.open(fr, 'r', encoding='utf-8')
    relDict = {}
    lenDict = {}
    max_s = 0
    for line in frr.readlines():
        # print(line)
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        words = jline['sent'].split(' ')
        e1_posi = jline['e1_posi']
        e2_posi = jline['e2_posi']

        max_long = max(e1_posi[1], e2_posi[1])
        if len(words) > 100 and max_long > 100:
            continue

        if len(words) not in lenDict.keys():
            lenDict[len(words)] = 0
        lenDict[len(words)] += 1

        max_s = max(max_s, len(words))
        print(max_s)
        rel = jline['rel']
        if rel in relDict.keys():
            relDict[rel] += 1
        else:
            relDict[rel] = 1
    print(len(relDict))
    frr.close()

    # lenlist = sorted(lenDict.items(), key=lambda m: m[0], reverse=False)
    # cc = 0
    # for ll in lenlist:
    #     cc += ll[1]
    #     print(ll, cc, cc/225060)

    rel4Test = []
    relList = list(relDict.keys())
    i = 0
    while i * 5 + 4 < len(relList):
        nd = random.randint(0, 4)
        rel4Test.append(relList[i * 5 + nd])
        i += 1

    print(len(rel4Test))
    print(rel4Test)
    #
    # frr = codecs.open(fr, 'r', encoding='utf-8')
    #
    # for line in frr.readlines():
    #     # print(line)
    #     jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
    #
    #     rel = jline['rel']
    #
    #     fj = json.dumps(jline, ensure_ascii=False)
    #     if rel in rel4Test:
    #         fwte.write(fj + '\n')
    #     else:
    #         fwtr.write(fj + '\n')
    #
    # fwte.close()
    # fwtr.close()
    frr.close()

    # relList = sorted(relDict.items(), key=lambda s: s[1], reverse=True)
    # for rr in relList:
    #     print(rr)
    # print(len(relList))


def get_rel2v_ave_glove100():

    fr = './data/WikiReading/WikiReading.txt.json.txt'
    frr = codecs.open(fr, 'r', encoding='utf-8')
    rellist = []
    for line in frr.readlines():
        # print(line)
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        rel = jline['rel']
        if rel not in rellist:
            rellist.append(rel)

    frr.close()


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
    # for ki in rellist:
    #     for w in ki.split():
    #         # print(w)
    #         i += 1
    #         if w not in w2v.keys():
    #             j += 1
    #             print(ki, w)
    # print(i, j)

    rel2v_file = "./data/WikiReading/WikiReading.rel2v.by_glove.100d.txt"
    fw = open(rel2v_file, 'w', encoding='utf-8')

    for ri, rel in enumerate(rellist):
        print(rel)
        words = rel.split()
        # print(words)
        wpos = nltk.pos_tag(words)
        newwords = []
        for wi, wp in enumerate(wpos):
            if wp[1] == 'IN' or wp[1] == 'TO':
                continue
            newwords.append(wp[0])
        words = newwords
        # print(words)
        W = np.zeros(shape=(words.__len__(), 100))
        for wi, w in enumerate(words):
            lower_word = w.lower()
            W[wi] = w2v[lower_word]
        ave = np.mean(W, axis=0)
        string = rel
        for item in ave:
            string += ' ' + str(item)
        fw.write(string + '\n')
    fw.close()


def find_rel_from_corpus():

    f = './data/WikiReading/'
    fw = codecs.open(f + 'WikiReading.2.txt', 'w', encoding='utf-8')
    fname = ['train.', 'dev.', 'test.']
    relDict = {}

    for name in fname:
        for fni in range(0, 10):
            print(name, fni)
            fr = codecs.open(f+name+str(fni), 'r', encoding='utf-8')
            lines = fr.readlines()
            for line in lines:
                lsp = line.rstrip('\n').split('\t')
                if len(lsp) < 5:
                    continue
                rel = lsp[0]
                if rel not in relDict.keys():
                    relDict[rel] = []
                val = lsp[2] + '\t' + lsp[3] + '\t' + lsp[4]
                if val not in relDict[rel]:
                    relDict[rel] += [val]
                    fw.write(line)
                # relDict[rel] += 1
            fr.close()
    fw.close()

    for ri, rel in enumerate(relDict.keys()):

        print(ri, rel, len(relDict[rel]))


def find_rel_in_test():

    fr = './data/WikiReading/WikiReading的副本.txt'
    frr = codecs.open(fr, 'r', encoding='utf-8')
    rellist = {}
    for line in frr.readlines():
        # print(line)
        jline = line.rstrip('\r\n').rstrip('\n').split('\t')
        rel = jline[0]
        ques = jline[1]

        if rel not in rellist:
            rellist[rel] = []
        if ques not in rellist[rel]:
            rellist[rel].append(ques)

    frr.close()
    for rr in rellist:
        for qq in rellist[rr]:
            print(rr + '\t'+ qq)


if __name__ == '__main__':


    get_rel2v_ave_glove100()

    # find_rel_from_corpus()

    # Process_Corpus()

    # Split_zeroshotData_2_train_test()

    # find_rel_in_test()




'''
0 place of birth 5147
1 nominated for 126
2 illustrator 332
3 mouth of the watercourse 1499
4 taxon rank 5211
5 dissolved or abolished 1672
6 noble family 449
7 member of sports team 4585
8 named after 1644
9 film editor 320
10 manufacturer 1501
11 location of formation 408
12 programming language 652
13 founder 1631
14 noble title 259
15 canonization status 387
16 located in the administrative territorial entity 5198
17 lyrics by 1050
18 characters 364
19 end time 1129
20 position held 3812
21 military rank 136
22 start time 684
23 parent taxon 4940
24 place of death 4843
25 airline hub 312
26 chromosome 191
27 distributor 722
28 designer 608
29 licensed to broadcast to 2201
30 instrumentation 156
31 record label 3268
32 brother 1131
33 editor 118
34 located next to body of water 270
35 voice type 2327
36 screenwriter 3805
37 conferred by 221
38 member of political party 3873
39 publication date 5032
40 position played on team / speciality 4769
41 continent 3305
42 language of work or name 956
43 performer 4993
44 headquarters location 4476
45 present in work 1677
46 from fictional universe 632
47 inception 4930
48 stock exchange 581
49 country of citizenship 5112
50 found in taxon 576
51 publisher 3254
52 father 2953
53 mother 1377
54 sport 4882
55 languages spoken or written 4917
56 crosses 598
57 country of origin 3282
58 instrument 1383
59 author 4737
60 league 555
61 architectural style 112
62 architect 601
63 sister 247
64 native language 3815
65 connecting line 1093
66 date of birth 5231
67 discoverer or inventor 662
68 place of burial 541
69 chairperson 195
70 time of discovery 133
71 located on astronomical body 521
72 creator 2815
73 vessel class 757
74 religious order 229
75 site of astronomical discovery 221
76 participant of 4156
77 child 1456
78 developer 2566
79 license 245
80 operating system 187
81 time of spacecraft launch 349
82 country 5234
83 cast member 4567
84 spouse 2013
85 based on 513
86 occupation 5208
87 military branch 2737
88 educated at 3438
89 date of death 5159
90 production company 982
91 standards body 304
92 collection 421
93 medical condition 108
94 convicted of 234
95 drafted by 444
96 constellation 2587
97 parent company 616
98 service entry 227
99 sex or gender 4054
100 original network 3190
101 narrative location 1086
102 residence 688
103 series 3504
104 conflict 2726
105 director 4874
106 award received 3314
107 point in time 1017
108 home venue 141
109 IUCN conservation status 426
110 material used 752
111 replaced by 175
112 product 184
113 industry 477
114 cause of death 1036
115 occupant 608
116 date of official opening 129
117 head of government 136
118 manner of death 235
119 employer 3276


'''