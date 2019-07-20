# -*- coding: UTF-8 -*-
import codecs, json
import re, nltk,random


def Process_Corpus():
    f = './data/annotated_fb_data_zeroshot_2.txt'
    # fw1w = codecs.open(f + '.2.txt', 'w', encoding='utf-8')
    # frr = codecs.open(f, 'r', encoding='utf-8')
    # jsondict = {}
    # for line in frr.readlines():
    #     ls = line.rstrip('\n').split('\t')
    #     jsondict['sent'] = ls[5]
    #     jsondict['en1_name'] = ls[0]
    #     jsondict['en1_id'] = ls[2]
    #     jsondict['en2_name'] = ls[1]
    #     jsondict['en2_id'] = ls[4]
    #     jsondict['rel'] = ls[3]
    #
    #     posilist = nltk.pos_tag(nltk.word_tokenize(ls[5]))
    #     jsondict['pos'] = posilist
    #     # print(posilist)
    #     fj = json.dumps(jsondict, ensure_ascii=False)
    #     fw1w.write(fj + '\n')
    #     # fw1w.write(line.rstrip('\n')+'\n')
    #
    #
    # frr.close()
    # fw1w.close()

    frr2 = codecs.open(f + '.2.txt', 'r', encoding='utf-8')
    for line in frr2.readlines():

        sent = json.loads(line.strip('\r\n').strip('\n'))
        print(sent['sent'])
        poslist = sent['pos']
        for pos in poslist:
            print(pos[1])


def analysis_entity_type():

    fty = '/Users/shengbinjia/Downloads/FB5M-extra/FB5M.type.txt'
    frty = codecs.open(fty, 'r', encoding='utf-8')

    entDict = {}
    typeDict = {}
    line = frty.readline()
    while line:
        sp = line.rstrip('\n').split('\t')
        if sp[0] not in entDict.keys():
            entDict[sp[0]] = 1
        else:
            entDict[sp[0]] += 1

        if sp[2] not in typeDict.keys():
            typeDict[sp[2]] = 1
        else:
            typeDict[sp[2]] += 1

        line = frty.readline()

    print(len(entDict), len(typeDict))

    entlist = sorted(entDict.items(), key=lambda x: x[1], reverse=False)
    print(entlist)

    typelist = sorted(typeDict.items(), key=lambda x: x[1], reverse=False)
    print(typelist)
    




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




def Ques2Sent():

    fr = './data/annotated_fb_data_zeroshot_2.txt'
    fw1 = './data/annotated_fb__zeroshot_RE.txt'
    fw2 = './data/annotated_fb_data_zeroshot_3.txt'
    fw1w = codecs.open(fw1, 'w', encoding='utf-8')
    fw2w = codecs.open(fw2, 'w', encoding='utf-8')
    count1w = 0
    count2w = 0
    cc = 0
    max_s = 0
    frr = codecs.open(fr, 'r', encoding='utf-8')
    for line in frr.readlines():
        # print(line)
        jline = json.loads(line.rstrip('\r\n').rstrip('\n'))
        ques = jline['ques']
        en1_name0 = jline['e1_name']
        en2_name0 = jline['e2_name']
        poslist = jline['ques_pos']
        max_s = max(max_s, len(poslist))
        # print(max_s)

        e1_l = -1
        e1_r = -1
        e2_l = -1
        e2_r = -1

        matchObj = re.match(r'Name ', ques)
        if matchObj != None:# name is ...
            count1w += 1
            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)
            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent = re.sub(r'Name ', en2_name0 + ' is ', sent)
            else:
                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent = re.sub(r'Name ', en1_name0 + ' is ', sent)
                else:
                    cc += 1
                    # print(ques)
                    # print(sent)
                    # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue


        matchObj = re.match(r'((Whats)|(What\'s)) ', ques)
        if matchObj != None:
            count1w += 1
            sent = ''
            
            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)
            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)
            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent = re.sub(r'((Whats)|(What\'s)) ', en2_name0 + ' is ', sent)
            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent = re.sub(r'((Whats)|(What\'s)) ', en1_name0 + ' is ', sent)
                else:
                    cc += 1
                    # print(sent)
                    # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)
            continue
        

        matchObj = re.match(r'^((What)|(Which)) ((are)|(is)|(was)|(were)) ', ques)
        if matchObj != None:
            count1w += 1
            sent = ''
            
            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)
            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)
            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent = re.sub(r'^((What)|(Which)) ', en2_name0 + ' ', sent)
            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent = re.sub(r'^((What)|(Which)) ', en1_name0 + ' ', sent)
                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)
            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)
            continue


        matchObj0 = re.match(r'Where ((are)|(is)|(was)|(were)) ', ques)
        if matchObj0 != None:
            count1w += 1
            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group()

                wl, wr = re.search(r'Where ((are)|(is)|(was)|(were)) ', ques, flags=re.I).span()
                el, er = re.search(en1_name, ques, flags=re.I).span()
                if wr == el and len(ques) == er:
                    sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', '', sent)
                    sent += ' is in ' + en2_name0

                elif wr != el and len(ques) == er:
                    sent_group = re.sub(r'Where ', en2_name0 + ' ', sent_group)
                    sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', sent_group, sent)

                else:
                    sent_group = re.sub(r'Where ', en2_name0 + ' ', sent_group) + 'where '
                    sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', sent_group, sent)
                    # print(sent)
                    # print(line)

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group()

                    wl, wr = re.search(r'Where ((are)|(is)|(was)|(were)) ', ques, flags=re.I).span()
                    el, er = re.search(en2_name, ques, flags=re.I).span()
                    if wr == el and len(ques) == er:
                        sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', '', sent)
                        sent += ' is in ' + en1_name0

                    elif wr != el and len(ques) == er:
                        sent_group = re.sub(r'Where ', en1_name0 + ' ', sent_group)
                        sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', sent_group, sent)

                    else:
                        sent_group = re.sub(r'Where ', en1_name0 + ' ', sent_group) + 'where '
                        sent = re.sub(r'Where ((are)|(is)|(was)|(were)) ', sent_group, sent)
                        # print(sent)
                        # print(line)
                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)
            continue


        matchObj0 = re.match(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', ques)
        if matchObj0 != None:
            count1w += 1
            # print(line)

            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group()

                wl, wr = re.search(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', ques, flags=re.I).span()
                el, er = re.search(en1_name, ques, flags=re.I).span()
                if wr == el and len(ques) == er:
                    sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', '', sent)
                    sent += ' is in ' + en2_name0

                elif wr != el and len(ques) == er:
                    sent_group = re.sub(r'Where ', en2_name0 + ' ', sent_group)
                    sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', sent_group, sent)

                else:
                    sent_group = re.sub(r'Where ', en2_name0 + ' ', sent_group) + 'where '
                    sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', sent_group, sent)

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group()

                    wl, wr = re.search(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', ques, flags=re.I).span()
                    el, er = re.search(en2_name, ques, flags=re.I).span()
                    if wr == el and len(ques) == er:
                        sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', '', sent)
                        sent += ' is in ' + en1_name0

                    elif wr != el and len(ques) == er:
                        sent_group = re.sub(r'Where ', en1_name0 + ' ', sent_group)
                        sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', sent_group, sent)

                    else:

                        sent_group = re.sub(r'Where ', en1_name0 + ' ', sent_group) + 'where '
                        sent = re.sub(r'Where in ([\w ]+) ((are)|(is)|(was)|(were)) ', sent_group, sent)
                        # print(sent)
                        # print(line)
                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)
            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)
            continue


        matchObj0 = re.match(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) of ', ques)
        if matchObj0 != None:
            count1w += 1
            # print(line)

            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group().split(' ')
                sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) of ',
                              'The ' + str(sent_group[1]) + ' ' + en2_name0 + ' ' + str(sent_group[2]) + ' of ', sent)

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group().split(' ')
                    sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) of ',
                                  'The ' + str(sent_group[1]) + ' ' + en1_name0 + ' ' + str(sent_group[2]) + ' of ', sent)
                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)

            # print(sent)
            # print(line)

            if sent != '':
                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue


        matchObj00 = re.match(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) of ', ques)
        matchObj0 = re.match(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ', ques)
        if matchObj00 == None and matchObj0 != None:
            count1w += 1

            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group().split(' ')
                if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                    el, er = re.search(en1_name, sent).span()
                    sent = sent[:er] + ' ' + str(sent_group[2]) + sent[er:] + ' the ' + str(sent_group[1]) + ' ' + en2_name0
                    sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                      '', sent)

                elif 'VB' in poslist[-1][1]:
                    sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                  'The ' + str(sent_group[1]) + ' that ', sent)
                    sent += ' ' + str(sent_group[2]) + ' ' + en2_name0

                elif 'IN' == poslist[3][1] or 'TO' == poslist[3][1] or 'VB' in poslist[3][1]:
                    sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                  'The ' + str(sent_group[1]) + ' ' + en2_name0 + ' ' + str(sent_group[2]) + ' ', sent)

                else:
                    sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                  'The ' + str(sent_group[1]) + ' of ', sent)
                    sent += ' ' + str(sent_group[2]) + ' ' + en2_name0

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group().split(' ')

                    if 'IN' == poslist[3][1] or 'TO' == poslist[3][1] or 'VB' in poslist[3][1]:
                        sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                      'The ' + str(sent_group[1]) + ' ' + en1_name0 + ' ' + str(sent_group[2]) + ' ',
                                      sent)

                    else:
                        sent = re.sub(r'((What)|(Which)) (\w+) ((are)|(is)|(was)|(were)) ',
                                      'The ' + str(sent_group[1]) + ' of ', sent)
                        sent += ' ' + str(sent_group[2]) + ' ' + en1_name0

                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)

            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue


        matchObj0 = re.match(r'((What)|(Which)) (\w+) ((do)|(does)|(did)) ', ques)
        if matchObj0 != None:
            count1w += 1
            # print(line)
            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group().split(' ')

                if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                    sent = sent + ' the ' + str(sent_group[1]) + ' ' + en2_name0
                    sent = re.sub(r'((What)|(Which)) (\w+) ((do)|(does)|(did)) ', '', sent)

                else:
                    sent = re.sub(r'((What)|(Which)) (\w+) ((do)|(does)|(did)) ',
                                  'The ' + str(sent_group[1]) + ' that ', sent)
                    sent += ' is the ' + en2_name0

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group().split(' ')

                    if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                        sent = sent + ' the ' + str(sent_group[1]) + ' ' + en1_name0
                        sent = re.sub(r'((What)|(Which)) (\w+) ((do)|(does)|(did)) ', '', sent)

                    else:
                        sent = re.sub(r'((What)|(Which)) (\w+) ((do)|(does)|(did)) ',
                                      'The ' + str(sent_group[1]) + ' that ', sent)
                        sent += ' is the ' + en1_name0

                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)

            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue


        matchObj0 = re.match(r'Where ((do)|(does)|(did)) ', ques)
        if matchObj0 != None: #has prep or no

            count1w += 1
            # print(line)
            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group().split(' ')

                if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                    sent = sent + ' ' + en2_name0
                    sent = re.sub(r'Where ((do)|(does)|(did)) ', '', sent)

                else:
                    sent = re.sub(r'Where ((do)|(does)|(did)) ',
                                  'Where ', sent)
                    sent += ' is ' + en2_name0

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group().split(' ')

                    if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                        sent = sent + ' ' + en1_name0
                        sent = re.sub(r'Where ((do)|(does)|(did)) ', '', sent)

                    else:
                        sent = re.sub(r'Where ((do)|(does)|(did)) ', 'Where ', sent)
                        sent += ' is ' + en1_name0

                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)

            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue


        matchObj0 = re.match(r'Where in ([\S ]+) ((do)|(does)|(did)) ', ques)
        if matchObj0 != None: #has prep or no

            count1w += 1
            # print(line)

            sent = ''

            en1_name = re.sub(r'\(', '\\\(', en1_name0)
            en1_name = re.sub(r'\)', '\\\)', en1_name)

            en2_name = re.sub(r'\(', '\\\(', en2_name0)
            en2_name = re.sub(r'\)', '\\\)', en2_name)

            matchObj = re.search(en1_name, ques, flags=re.I)
            if matchObj != None:
                pattern = re.compile(en1_name, flags=re.I)
                sent = re.sub(pattern, en1_name0, ques)
                sent_group = matchObj0.group().split(' ')

                if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                    sent = sent + ' ' + en2_name0
                    sent = re.sub(r'Where in ([\S ]+) ((do)|(does)|(did)) ', '', sent)

                else:
                    sent = re.sub(r'Where in ([\S ]+) ((do)|(does)|(did)) ', 'Where ', sent)
                    sent += ' is ' + en2_name0

            else:

                matchObj = re.search(en2_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en2_name, flags=re.I)
                    sent = re.sub(pattern, en2_name0, ques)
                    sent_group = matchObj0.group().split(' ')

                    if 'IN' == poslist[-1][1] or 'TO' == poslist[-1][1]:
                        sent = sent + ' ' + en1_name0
                        sent = re.sub(r'Where in ([\S ]+) ((do)|(does)|(did)) ', '', sent)

                    else:
                        sent = re.sub(r'Where in ([\S ]+) ((do)|(does)|(did)) ', 'Where ', sent)
                        sent += ' is ' + en1_name0

                else:
                    cc += 1
                    # print(cc, sent)
                    # print(line)
            # print(sent)
            # print(line)

            if sent != '':

                e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                # print(e1_l, e1_r, e2_l, e2_r)

                if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                    print('>>>>>>>>>>>>>\n', sent)
                    print(poslist)
                    print('e1 taken.....', nltk.word_tokenize(en1_name0))
                    print('e2 taken.....', nltk.word_tokenize(en2_name0))
                    print(line)
                    print(e1_l, e1_r, e2_l, e2_r)
                else:
                    if e1_r >= e2_l or e2_r >= e1_l:
                        jline['sent'] = sent
                        jline['e1_posi'] = (e1_l, e1_r)
                        jline['e2_posi'] = (e2_l, e2_r)
                        fj = json.dumps(jline, ensure_ascii=False)
                        fw1w.write(fj + '\n')
                    else:
                        print(line)

            continue

        
        matchObj = re.match(r'^Who ', ques)
        if matchObj != None:
            if 'VB' in poslist[1][1] or 'MD' == poslist[1][1]:
                count1w += 1
                # print(line)
                sent = ''

                en1_name = re.sub(r'\(', '\\\(', en1_name0)
                en1_name = re.sub(r'\)', '\\\)', en1_name)
                en2_name = re.sub(r'\(', '\\\(', en2_name0)
                en2_name = re.sub(r'\)', '\\\)', en2_name)
                matchObj = re.search(en1_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en1_name, flags=re.I)
                    sent = re.sub(pattern, en1_name0, ques)
                    sent = re.sub(r'^Who ', en2_name0 + ' ', sent)
                    # print(sent)
                    # print(line)
                else:

                    matchObj = re.search(en2_name, ques, flags=re.I)
                    if matchObj != None:
                        pattern = re.compile(en2_name, flags=re.I)
                        sent = re.sub(pattern, en2_name0, ques)
                        sent = re.sub(r'^Who ', en1_name0 + ' ', sent)

                    else:
                        cc += 1
                        # print(cc, sent)
                        # print(line)
                # print(sent)
                # print(line)

                if sent != '':

                    e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                    e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                    e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                    e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                    e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                    e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                    # print(e1_l, e1_r, e2_l, e2_r)

                    if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                        print('>>>>>>>>>>>>>\n', sent)
                        print(poslist)
                        print('e1 taken.....', nltk.word_tokenize(en1_name0))
                        print('e2 taken.....', nltk.word_tokenize(en2_name0))
                        print(line)
                        print(e1_l, e1_r, e2_l, e2_r)
                    else:
                        if e1_r >= e2_l or e2_r >= e1_l:
                            jline['sent'] = sent
                            jline['e1_posi'] = (e1_l, e1_r)
                            jline['e2_posi'] = (e2_l, e2_r)
                            fj = json.dumps(jline, ensure_ascii=False)
                            fw1w.write(fj + '\n')
                        else:
                            print(line)

                continue

            elif 'RB' in poslist[1][1] and ('VB' in poslist[2][1] or 'MD' == poslist[2][1]):
                count1w += 1
                # print(line)
                sent = ''

                en1_name = re.sub(r'\(', '\\\(', en1_name0)
                en1_name = re.sub(r'\)', '\\\)', en1_name)
                en2_name = re.sub(r'\(', '\\\(', en2_name0)
                en2_name = re.sub(r'\)', '\\\)', en2_name)
                matchObj = re.search(en1_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en1_name, flags=re.I)
                    sent = re.sub(pattern, en1_name0, ques)
                    sent = re.sub(r'^Who ', en2_name0 + ' ', sent)
                    # print(sent)
                    # print(line)
                else:

                    matchObj = re.search(en2_name, ques, flags=re.I)
                    if matchObj != None:
                        pattern = re.compile(en2_name, flags=re.I)
                        sent = re.sub(pattern, en2_name0, ques)
                        sent = re.sub(r'^Who ', en1_name0 + ' ', sent)
                        # print(sent)
                        # print(line)
                    else:
                        cc += 1
                        # print(cc, sent)
                        # print(line)
                # print(sent)
                # print(line)

                if sent != '':

                    e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                    e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                    e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                    e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                    e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                    e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                    # print(e1_l, e1_r, e2_l, e2_r)

                    if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                        print('>>>>>>>>>>>>>\n', sent)
                        print(poslist)
                        print('e1 taken.....', nltk.word_tokenize(en1_name0))
                        print('e2 taken.....', nltk.word_tokenize(en2_name0))
                        print(line)
                        print(e1_l, e1_r, e2_l, e2_r)
                    else:
                        if e1_r >= e2_l or e2_r >= e1_l:
                            jline['sent'] = sent
                            jline['e1_posi'] = (e1_l, e1_r)
                            jline['e2_posi'] = (e2_l, e2_r)
                            fj = json.dumps(jline, ensure_ascii=False)
                            fw1w.write(fj + '\n')
                        else:
                            print(line)

                continue

            elif 'IN' in poslist[1][1]:
                count1w += 1
                # print(line)
                sent = ''

                en1_name = re.sub(r'\(', '\\\(', en1_name0)
                en1_name = re.sub(r'\)', '\\\)', en1_name)
                en2_name = re.sub(r'\(', '\\\(', en2_name0)
                en2_name = re.sub(r'\)', '\\\)', en2_name)
                matchObj = re.search(en1_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en1_name, flags=re.I)
                    sent = re.sub(pattern, en1_name0, ques)
                    sent = re.sub(r'^Who ', en2_name0 + ' ', sent)
                    # print(sent)
                    # print(line)
                else:

                    matchObj = re.search(en2_name, ques, flags=re.I)
                    if matchObj != None:
                        pattern = re.compile(en2_name, flags=re.I)
                        sent = re.sub(pattern, en2_name0, ques)
                        sent = re.sub(r'^Who ', en1_name0 + ' ', sent)
                        # print(sent)
                        # print(line)
                    else:
                        cc += 1
                        # print(cc, sent)
                        # print(line)
                # print(sent)
                # print(line)

                if sent != '':

                    e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                    e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                    e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                    e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                    e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                    e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                    # print(e1_l, e1_r, e2_l, e2_r)

                    if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                        print('>>>>>>>>>>>>>\n', sent)
                        print(poslist)
                        print('e1 taken.....', nltk.word_tokenize(en1_name0))
                        print('e2 taken.....', nltk.word_tokenize(en2_name0))
                        print(line)
                        print(e1_l, e1_r, e2_l, e2_r)
                    else:
                        if e1_r >= e2_l or e2_r >= e1_l:
                            jline['sent'] = sent
                            jline['e1_posi'] = (e1_l, e1_r)
                            jline['e2_posi'] = (e2_l, e2_r)
                            fj = json.dumps(jline, ensure_ascii=False)
                            fw1w.write(fj + '\n')
                        else:
                            print(line)

                continue


        matchObj = re.match(r'^((Which)|(What)) ', ques)
        if matchObj != None:
            if 'VB' in poslist[2][1] and 'NN' in poslist[1][1]:
                count1w += 1
                # print(line)

                sent = ''

                en1_name = re.sub(r'\(', '\\\(', en1_name0)
                en1_name = re.sub(r'\)', '\\\)', en1_name)

                en2_name = re.sub(r'\(', '\\\(', en2_name0)
                en2_name = re.sub(r'\)', '\\\)', en2_name)

                matchObj = re.search(en1_name, ques, flags=re.I)
                if matchObj != None:
                    pattern = re.compile(en1_name, flags=re.I)
                    sent = re.sub(pattern, en1_name0, ques)
                    sent = re.sub(r'^((What)|(Which)) (\S+) ',
                                      'The ' + str(poslist[1][0]) + ' ' + en2_name0 + ' ', sent)

                else:

                    matchObj = re.search(en2_name, ques, flags=re.I)
                    if matchObj != None:
                        pattern = re.compile(en2_name, flags=re.I)
                        sent = re.sub(pattern, en2_name0, ques)

                        sent = re.sub(r'^((What)|(Which)) (\S+) ',
                                      'The ' + str(poslist[1][0]) + ' ' + en1_name0 + ' ', sent)
                        # print(sent)
                        # print(line)

                    else:
                        cc += 1
                        # print(cc, sent)
                        # print(line)

                # print(sent)
                # print(line)

                if sent != '':
                    e1_0, e1_1 = re.search(en1_name, sent, flags=re.I).span()
                    e1_l = len(re.compile(r' ').findall(sent[:e1_0]))
                    e1_r = len(re.compile(r' ').findall(sent[:e1_1])) + 1
                    e2_0, e2_1 = re.search(en2_name, sent, flags=re.I).span()
                    e2_l = len(re.compile(r' ').findall(sent[:e2_0]))
                    e2_r = len(re.compile(r' ').findall(sent[:e2_1])) + 1
                    # print(e1_l, e1_r, e2_l, e2_r)

                    if e1_l == -1 or e1_r == -1 or e2_l == -1 or e2_r == -1:
                        print('>>>>>>>>>>>>>\n', sent)
                        print(poslist)
                        print('e1 taken.....', nltk.word_tokenize(en1_name0))
                        print('e2 taken.....', nltk.word_tokenize(en2_name0))
                        print(line)
                        print(e1_l, e1_r, e2_l, e2_r)
                    else:
                        if e1_r >= e2_l or e2_r >= e1_l:
                            jline['sent'] = sent
                            jline['e1_posi'] = (e1_l, e1_r)
                            jline['e2_posi'] = (e2_l, e2_r)
                            fj = json.dumps(jline, ensure_ascii=False)
                            fw1w.write(fj + '\n')
                        else:
                            print(line)

                continue



        fw2w.write(line)
        count2w += 1

    frr.close()
    fw1w.close()
    fw2w.close()
    print(count1w, count2w, count1w+count2w)


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

    # Process_Corpus()

    # review_ents()

    # Split_zeroshotData_2_train_test()

    analysis_entity_type()

    # print(re.search('www', 'www.runoobwww.com').group())  # 在起始位置匹配
    # print(re.match('com', 'www.runoob.com'))  # 不在起始位置匹配
    #
    # line = "Cats are smarter than dogs"
    #
    # matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)
    #
    # if matchObj:
    #     print("matchObj.group() : ", matchObj.group())
    #     print("matchObj.group(1) : ", matchObj.group(1))
    #     print("matchObj.group(3) : ", matchObj.group(0))
    # else:
    #     print("No match!!")
    #
    # s = 'What group includes flemish people'
    # s = 'What county and state is  holloway from'
    # matchObj = re.match(r'What (\w+) (are)|(is)|(was)|(were)', s)
    #
    # print(matchObj)

