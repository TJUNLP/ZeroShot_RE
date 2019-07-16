
def test_count_of_coupus():
    f = '/Users/shengbinjia/Downloads/KRL_FB15K20K'
    fr1 = f + '/FB15K/relation2id.txt'
    fr2 = '/Users/shengbinjia/Downloads/SimpleQuestions_v2/annotated_fb_data_original.txt'
    fzeroshot = '/Users/shengbinjia/Downloads/SimpleQuestions_v2/annotated_fb_data_zeroshot.txt'

    fw = open(fzeroshot, 'w')
    fr = open(fr1, 'r')
    line = fr.readline()
    middict = {}
    while line:
        ls = line.rstrip('\n').split('\t')
        if ls[0] not in middict:
            middict[ls[0]] = 0
        line = fr.readline()

    fr.close()
    print(len(middict))

    fr = open(fr2, 'r')
    line = fr.readline()
    count = 0
    while line:
        ls = line.rstrip('\n').split('\t')
        mid = ls[1]
        if mid in middict.keys():
            middict[mid] += 1
            fw.write(line)
            count += 1

        line = fr.readline()
    print(count)
    fr.close()
    fw.close()
    midlist = sorted(middict.items(), key=lambda d: d[1], reverse=True)
    num = 0
    for md in midlist:

        # print(md)
        if md[1] != 0:
            num += 1
    print(num)


def test_ent_exit():
    f1 = '/Users/shengbinjia/Downloads/FB5M-extra/FB5M.en-name.txt'
    fzeroshot = '/Users/shengbinjia/Downloads/SimpleQuestions_v2/annotated_fb_data_zeroshot.txt'
    fzeroshot2 = '/Users/shengbinjia/Downloads/SimpleQuestions_v2/annotated_fb_data_zeroshot_2.txt'

    namedict = {}
    fr = open(f1, 'r')
    line = fr.readline()
    while line:
        ls = line.rstrip('\n').split('\t')
        if '<fb:type.object.en_name>' != ls[1]:
            line = fr.readline()
            continue

        na = '/m/'+ls[0][6:-1]
        if na not in namedict:
            namedict[na] = ls[2].strip('"')
        else:
            print(line)
        line = fr.readline()

    fr.close()

    # for nd in namedict.keys():
    #     print(nd, namedict[nd])
    print(len(namedict))

    count = 0
    fw = open(fzeroshot2, 'w')
    fr = open(fzeroshot, 'r')

    for id, line in enumerate(fr.readlines()):
        print(id)
        ls = line.rstrip('\n').split('\t')
        strstr = ''
        if ls[0] in namedict.keys():
            strstr += namedict[ls[0]] + '\t'
        else:
            print(ls[0], line)
            count += 1
            continue
        if ls[2] in namedict.keys():
            strstr += namedict[ls[2]] + '\t'
        else:
            print(ls[2], line)
            count += 1
            continue

        fw.write(strstr + line)

    fw.close()
    fr.close()
    print(count)


if __name__ == '__main__':

    # test_count_of_coupus()
    test_ent_exit()


