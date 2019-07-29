
import tensorflow as k
import codecs, json,random
import numpy as np


def Process_Corpus():

    f = './data/WikiReading/WikiReading.txt'
    fw1w = codecs.open(f + '.json.txt', 'w', encoding='utf-8')
    frr = codecs.open(f, 'r', encoding='utf-8')
    jsondict = {}
    for line in frr.readlines():
        ls = line.rstrip('\n').split('\t')
        jsondict['sent'] = ls[3]
        assert ls[2] in ls[3]
        assert ls[4] in ls[3]
        jsondict['en1_name'] = ls[2]
        jsondict['en2_name'] = ls[4]
        jsondict['rel'] = ls[0]

        fj = json.dumps(jsondict, ensure_ascii=False)
        fw1w.write(fj + '\n')
        # fw1w.write(line.rstrip('\n')+'\n')


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

    lenlist = sorted(lenDict.items(), key=lambda m: m[0], reverse=False)
    cc = 0
    for ll in lenlist:
        cc += ll[1]
        print(ll, cc, cc/386531)

    # rel4Test = []
    # relList = list(relDict.keys())
    # i = 0
    # while i * 5 + 4 < len(relList):
    #     nd = random.randint(0, 4)
    #     rel4Test.append(relList[i * 5 + nd])
    #     i += 1
    #
    # print(len(rel4Test))
    # print(rel4Test)
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
    fw = codecs.open(f + 'WikiReading.txt', 'w', encoding='utf-8')
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
                if line not in relDict[rel]:
                    relDict[rel] += [line]
                    fw.write(line)
                # relDict[rel] += 1
            fr.close()
    fw.close()

    for ri, rel in enumerate(relDict.keys()):

        print(ri, rel, len(relDict[rel]))



if __name__ == '__main__':

    # Process_Corpus()

    get_rel2v_ave_glove100()

    # find_rel_from_corpus()

    # Process_Corpus()

    # Split_zeroshotData_2_train_test()

'''
0 place of birth 5264
1 nominated for 252
2 illustrator 1294
3 mouth of the watercourse 4469
4 taxon rank 5247
5 dissolved or abolished 4500
6 noble family 1315
7 member of sports team 5219
8 named after 3694
9 film editor 3419
10 manufacturer 4719
11 location of formation 2420
12 programming language 1833
13 founder 4536
14 noble title 517
15 canonization status 387
16 located in the administrative territorial entity 5237
17 lyrics by 4302
18 characters 364
19 end time 1129
20 position held 4730
21 military rank 272
22 start time 684
23 parent taxon 5196
24 place of death 5176
25 airline hub 2379
26 chromosome 2076
27 distributor 4440
28 designer 3529
29 licensed to broadcast to 3766
30 instrumentation 2106
31 record label 4658
32 brother 2068
33 editor 472
34 located next to body of water 808
35 voice type 4663
36 screenwriter 5143
37 conferred by 221
38 member of political party 5034
39 publication date 5268
40 position played on team / speciality 5262
41 continent 5102
42 language of work or name 3941
43 performer 5258
44 headquarters location 5126
45 present in work 4556
46 from fictional universe 3799
47 inception 5262
48 stock exchange 3309
49 country of citizenship 5269
50 found in taxon 1653
51 publisher 4875
52 father 5045
53 mother 4300
54 sport 5251
55 languages spoken or written 5241
56 crosses 2141
57 country of origin 4850
58 instrument 4890
59 author 5251
60 league 3085
61 architectural style 336
62 architect 4497
63 sister 494
64 native language 5123
65 connecting line 4237
66 date of birth 5270
67 discoverer or inventor 4096
68 place of burial 3618
69 chairperson 195
70 time of discovery 2319
71 located on astronomical body 3165
72 creator 4747
73 vessel class 2488
74 religious order 458
75 site of astronomical discovery 1856
76 participant of 5106
77 child 2491
78 developer 4814
79 license 245
80 operating system 187
81 time of spacecraft launch 3148
82 country 5274
83 cast member 5189
84 spouse 2013
85 based on 1018
86 occupation 5237
87 military branch 5130
88 educated at 4977
89 date of death 5247
90 production company 3231
91 standards body 304
92 collection 421
93 medical condition 1815
94 convicted of 702
95 drafted by 2545
96 constellation 5078
97 parent company 1727
98 service entry 2190
99 sex or gender 5217
100 original network 5164
101 narrative location 2002
102 residence 1351
103 series 5117
104 conflict 5105
105 director 5261
106 award received 4764
107 point in time 4291
108 home venue 2969
109 IUCN conservation status 2674
110 material used 1462
111 replaced by 525
112 product 1089
113 industry 477
114 cause of death 4738
115 occupant 608
116 date of official opening 1702
117 head of government 544
118 manner of death 935
119 employer 3276

'''