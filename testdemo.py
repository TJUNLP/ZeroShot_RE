
import tensorflow as k
import codecs, json

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


if __name__ == '__main__':

    Process_Corpus()