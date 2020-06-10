# -*- coding: UTF-8 -*-
import codecs, json
import re, nltk, random


def Process_Corpus():
    f = './data/tacred/test.json'
    # fw1w = codecs.open(f + '.2.txt', 'w', encoding='utf-8')
    frr = codecs.open(f, 'r', encoding='utf-8')

    for line in frr.readlines():
        ls = line.strip('\r\n').strip('\n')
        jn = json.loads(ls)
        rellist = []
        for sent in jn:
            print(sent)
            label = sent['label'].strip('\'')
            if label != 'NA':
                label = label.split(':')[1]
            text = sent['text']
            ents = sent['ents']
            e1 = ents[0]
            print(e1, e1[0], text[e1[1]:e1[2]])
            e2 = ents[1]
            ann = sent['ann']
            print(text)
            if label not in rellist:
                rellist.append(label)
        print(rellist, len(rellist))
    #     posilist = nltk.pos_tag(nltk.word_tokenize(ls[5]))
    #     jsondict['pos'] = posilist
    #     # print(posilist)
    #     fj = json.dumps(jsondict, ensure_ascii=False)
    #     fw1w.write(fj + '\n')
    #     # fw1w.write(line.rstrip('\n')+'\n')
    #
    #
    frr.close()
    # fw1w.close()

    # frr2 = codecs.open(f + '.2.txt', 'r', encoding='utf-8')
    # for line in frr2.readlines():
    #
    #     sent = json.loads(line.strip('\r\n').strip('\n'))
    #     print(sent['sent'])
    #     poslist = sent['pos']
    #     for pos in poslist:
    #         print(pos[1])


def analysis_entity_type():

    # enLists = []
    # f = './data/annotated_fb_data_zeroshot_2.txt'
    # frr = codecs.open(f, 'r', encoding='utf-8')
    # i = 0
    # for line in frr.readlines():
    #     jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
    #     en1_id = jline['e1_id']
    #     en2_id = jline['e2_id']
    #     i +=1
    #     print('...', i)
    #     if en1_id not in enLists:
    #         enLists.append(en1_id)
    #     if en2_id not in enLists:
    #         enLists.append(en2_id)
    #
    # frr.close()


    # fty = '/Users/shengbinjia/Downloads/FB5M-extra/FB5M.type.txt'
    fty = './data/annotated_fb_data_zeroshot_2.txt.FB5M_type.txt'
    frty = codecs.open(fty, 'r', encoding='utf-8')
    # ftyw = codecs.open(f + '.FB5M_type.txt', 'w', encoding='utf-8')
    entDict = {}
    typeDict = {}
    line = frty.readline()
    l = 0
    while line:
        l += 1
        print(l)
        sp = line.rstrip('\n').split('\t')

        if '<fb:common.topic>' == sp[2]:
            line = frty.readline()
            continue

        en = '/m/'+sp[0][6:-1]
        # if en not in enLists:
        #     line = frty.readline()
        #     continue
        # ty = sp[2].split('.')[0][4:]
        ty = sp[2][4:]
        # print(ty)
        if en not in entDict.keys():
            entDict[en] = []
        if ty not in entDict[en]:
            entDict[en].append(ty)

        # if ty not in typeDict.keys():
        #     typeDict[ty] = 1
        # else:
        #     typeDict[ty] += 1
        # ftyw.write(line)
        line = frty.readline()

    frty.close()
    # ftyw.close()

    # entlist = sorted(entDict.items(), key=lambda x: x[1], reverse=False)
    # print(entlist)
    #
    # typelist = sorted(typeDict.items(), key=lambda x: x[1], reverse=False)
    # print(typelist)

    # 3970427   500
    print(len(entDict), len(typeDict))

    # f = './data/annotated_fb_data_zeroshot_2.txt'
    # frr = codecs.open(f, 'r', encoding='utf-8')
    #
    # rel_ep_1_Dict = {}
    # rel_ep_2_Dict = {}
    # i = 0
    # for line in frr.readlines():
    #     jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
    #     en1_id = jline['e1_id']
    #     en2_id = jline['e2_id']
    #     rel = jline['rel']
    #     i += 1
    #     print(i)
    #     if en1_id not in entDict or en2_id not in entDict:
    #         print(en1_id, en2_id, rel)
    #         continue
    #     # en1_type = entDict[en1_id]
    #     # en2_type = entDict[en2_id]
    #     en1_type = [en1_id]
    #     en2_type = [en2_id]
    #
    #     assert rel_ep_1_Dict.keys() == rel_ep_2_Dict.keys()
    #     if rel not in rel_ep_1_Dict.keys():
    #         rel_ep_1_Dict[rel] = en1_type
    #     else:
    #         rel_ep_1_Dict[rel] += en1_type
    #     if rel not in rel_ep_2_Dict.keys():
    #         rel_ep_2_Dict[rel] = en2_type
    #     else:
    #         rel_ep_2_Dict[rel] += en2_type
    #
    # frr.close()
    #
    # print(len(rel_ep_1_Dict), len(rel_ep_2_Dict))
    #
    # fw = codecs.open('./data/rel_entpair_Dict.txt','w', encoding='utf=8')
    #
    # assert rel_ep_1_Dict.keys() == rel_ep_2_Dict.keys()
    # for reli in rel_ep_1_Dict.keys():
    #     if reli not in rel_ep_2_Dict.keys():
    #         continue
    #     dicts = {}
    #     dicts['rel'] = reli
    #
    #     print(rel_ep_1_Dict[reli])
    #
    #     dicts['e1_types'] = rel_ep_1_Dict[reli]
    #     dicts['e2_types'] = rel_ep_2_Dict[reli]
    #
    #     fj = json.dumps(dicts, ensure_ascii=False)
    #
    #     fw.write(fj + '\n')
    #
    # fw.close()

    f = './data/rel_entpair_Dict.txt'
    frr = codecs.open(f, 'r', encoding='utf-8')
    fw = codecs.open('./data/rel_entpair_type_Dict.txt', 'w', encoding='utf=8')

    i = 0
    for line in frr.readlines():
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        en1_ids = list(jline['e1_types'])
        en2_ids = list(jline['e2_types'])
        rel = jline['rel']
        i += 1
        print(i)

        dicts = {}

        dicts['rel'] = rel

        en1_types = []
        for e1id in en1_ids:
            print(e1id)
            if e1id not in entDict:
                continue
            for ty in entDict[e1id]:
                if ty not in en1_types:
                    en1_types.append(ty)

        dicts['e1_types'] = en1_types

        en2_types = []
        for e2id in en2_ids:
            if e2id not in entDict:
                continue
            for ty in entDict[e2id]:
                if ty not in en2_types:
                    en2_types.append(ty)

        dicts['e2_types'] = en2_types

        fj = json.dumps(dicts, ensure_ascii=False)
        fw.write(fj + '\n')

    frr.close()
    fw.close()


def review_ents():
    fr = './data/annotated_fb_data_zeroshot_2.txt'
    frr2 = codecs.open(fr, 'r', encoding='utf-8')
    fw1w = codecs.open(fr + '.99.txt', 'w', encoding='utf-8')

    f1 = '/Users/shengbinjia/Downloads/FB5M-extra/FB5M.en-name.txt'
    namedict = {}
    frr1 = open(f1, 'r')
    line = frr1.readline()
    while line:
        ls = line.rstrip('\n').split('\t')

        na = '/m/'+ls[0][6:-1]
        if na not in namedict:
            namedict[na] = [ls[2].strip('"')]
        else:
            namedict[na].append(ls[2].strip('"'))
        line = frr1.readline()

    for name in namedict.keys():
        ll = sorted(namedict[name], key=lambda i: len(i), reverse=True)
        namedict[name] = ll
        # print(ll)

    frr1.close()

    count = 0
    for line in frr2.readlines():

        jline = json.loads(line.strip('\r\n').strip('\n'))
        ques = jline['ques']
        en1_name = jline['en1_name']
        en1_id = jline['en1_id']
        en2_name = jline['en2_name']
        en2_id = jline['en2_id']

        matchObj1 = None
        try:
            matchObj1 = re.search(en1_name, ques, flags=re.I)
        except BaseException:
            # print(en1_name, '!!!!!!!!!!!!----->>>', ques)
            pass
        matchObj2 = None
        try:
            matchObj2 = re.search(en2_name, ques, flags=re.I)
        except BaseException:
            # print(en1_name, '!!!!!!!!!!!!----->>>', ques)
            pass

        if matchObj1 == None and matchObj2 == None:

            for newname in namedict[en1_id]:
                try:
                    matchObj = re.search(newname, ques, flags=re.I)
                except BaseException:
                    pass
                else:
                    if matchObj != None:
                        count += 1
                        print(count, newname, '----->>>', ques, '--------', en1_id)
                        jline['en1_name'] = newname
                        break

            for newname in namedict[en2_id]:
                try:
                    matchObj = re.search(newname, ques, flags=re.I)
                except BaseException:
                    pass
                else:
                    if matchObj != None:
                        count += 1
                        print(count, newname, '----->>>', ques, '--------', en2_id)
                        jline['en2_name'] = newname
                        break


        fj = json.dumps(jline, ensure_ascii=False)
        fw1w.write(fj + '\n')

    fw1w.close()
    frr2.close()


def Split_zeroshotData_2_train_test():

    fw_train = './data/annotated_fb__zeroshot_RE.random.train.txt'
    fw_test = './data/annotated_fb__zeroshot_RE.random.test.txt'
    fwtr = codecs.open(fw_train, 'w', encoding='utf-8')
    fwte = codecs.open(fw_test, 'w', encoding='utf-8')

    fr = './data/annotated_fb__zeroshot_RE.txt'
    frr = codecs.open(fr, 'r', encoding='utf-8')
    relDict = {}
    max_s = 0
    for line in frr.readlines():
        # print(line)
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        words = jline['sent'].split(' ')
        max_s = max(max_s, len(words))
        # print(max_s)
        rel = jline['rel']
        if rel in relDict.keys():
            relDict[rel] += 1
        else:
            relDict[rel] = 1
    print(len(relDict))
    frr.close()

    rel4Test = []
    relList = list(relDict.keys())
    i = 0
    while i * 10 + 9 < len(relList):
        nd = random.randint(0, 9)
        rel4Test.append(relList[i * 10 + nd])
        i += 1

    print(len(rel4Test))
    print(rel4Test)

    frr = codecs.open(fr, 'r', encoding='utf-8')

    for line in frr.readlines():
        # print(line)
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        jline.pop('ques')
        jline.pop('ques_pos')

        rel = jline['rel']

        fj = json.dumps(jline, ensure_ascii=False)
        if rel in rel4Test:
            fwte.write(fj + '\n')
        else:
            fwtr.write(fj + '\n')

    fwte.close()
    fwtr.close()
    frr.close()

    # relList = sorted(relDict.items(), key=lambda s: s[1], reverse=True)
    # for rr in relList:
    #     print(rr)
    # print(len(relList))


def Write2file(sent, en1_name, en2_name):

    if sent != '':

        e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
        e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
        e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1

        e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
        e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
        e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1

        # print(e1_l, e1_r, e2_l, e2_r)

        # poslist = nltk.pos_tag(nltk.word_tokenize(sent))
        # e1_l = -1
        # e1_r = -1
        # e2_l = -1
        # e2_r = -1
        # item = 0
        # find_e1 = False
        # find_e2 = False
        # while item < len(poslist):
        #     # print(item, 'e1_l', e1_l)
        #     if (find_e1 == False) and (poslist[item][0] in nltk.word_tokenize(en1_name0)):
        #         e1_l = item
        #         e1_cname = []
        #         e1_cname.append(poslist[item][0])
        #         # print(e1_cname)
        #         j = item + 1
        #
        #         while j < len(poslist):
        #             if poslist[j][0] in nltk.word_tokenize(en1_name0):
        #                 e1_cname.append(poslist[j][0])
        #                 # print(e1_cname)
        #                 j += 1
        #             else:
        #                 # print(e1_cname_l, e1_cname_r)
        #                 if e1_cname == nltk.word_tokenize(en1_name0):
        #                     e1_r = j
        #                     find_e1 = True
        #                     item = j
        #                 else:
        #                     item = item + 1
        #                     e1_l = -1
        #                 break
        #
        #         if j == len(poslist):
        #             # print(e1_cname_l, e1_cname_r)
        #             if e1_cname == nltk.word_tokenize(en1_name0):
        #                 e1_r = j
        #                 find_e1 = True
        #                 item = j
        #             else:
        #                 item = j
        #                 e1_l = -1
        #     else:
        #         item += 1
        #
        # item = 0
        # while item < len(poslist):
        #     # print(item, 'e2_l', e2_l)
        #     if (find_e2 == False) and (poslist[item][0] in nltk.word_tokenize(en2_name0)):
        #         # print(item, '....e2_l', e2_l)
        #         e2_l = item
        #         e2_cname = [poslist[item][0]]
        #         # print(e1_cname)
        #         j = item + 1
        #
        #         while j < len(poslist):
        #             if poslist[j][0] in nltk.word_tokenize(en2_name0):
        #                 e2_cname.append(poslist[j][0])
        #                 # print(e1_cname)
        #                 j += 1
        #             else:
        #                 # print(e2_cname_l, e2_cname_r)
        #                 if e2_cname == nltk.word_tokenize(en2_name0):
        #                     e2_r = j
        #                     find_e2 = True
        #                     item = j
        #                 else:
        #                     item = item + 1
        #                     e2_l = -1
        #                 break
        #
        #         if j == len(poslist):
        #             # print(e1_cname_l, e1_cname_r)
        #             if e2_cname == nltk.word_tokenize(en2_name0):
        #                 e2_r = j
        #                 find_e2 = True
        #                 item = j
        #             else:
        #                 item = j
        #                 e2_l = -1
        #
        #     else:
        #         item += 1

        if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
            print('>>>>>>>>>>>>>\n', sent)
            # print(poslist)
            # print('e1 taken.....', nltk.word_tokenize(en1_name0))
            # print('e2 taken.....', nltk.word_tokenize(en2_name0))
            # print(line)
            print(e1_l, e1_r, e2_l, e2_r)


if __name__ == '__main__':

    print('---')

    # Ques2Sent()

    Process_Corpus()

    # review_ents()

    # Split_zeroshotData_2_train_test()

    # analysis_entity_type()



