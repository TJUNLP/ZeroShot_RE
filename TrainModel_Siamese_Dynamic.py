# -*- encoding:utf-8 -*-

# import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs, math
import os.path
import numpy as np
import ProcessData_Siamese
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_Siamese import Model_BiLSTM_sent_MLP__KGembed, Model_BiLSTM_sent_linear__KGembed


def test_model_4trainset(nn_model, pairs_test0, labels_test, classifer_labels_test, target_vob):

    data_s_all, data_tag_all, data_e1_posi_all, data_e2_posi_all, char_s_all = pairs_test0

    predict = 0
    predict_right = 0
    predict_right_c = 0
    predict_c = 0
    predict_right05 = 0
    totel_right = len(data_s_all)


    pairs_test = [data_s_all, data_tag_all, data_e1_posi_all, data_e2_posi_all, char_s_all]

    test_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test[3], dtype="int32")
    test_x1_sent_cahr = np.asarray(pairs_test[4], dtype="int32")

    predictions = nn_model.predict([test_x1_sent, test_x1_e1_posi, test_x1_e2_posi,
                                    test_x2_tag, test_x1_sent_cahr], batch_size=len(target_vob), verbose=0)


    for i in range(len(predictions)):

        predict += 1
        if labels_test[i] == 1 and predictions[i] > 0.5:
            predict_right += 1
            predict_right05 += 1


        if labels_test[i] == 0 and predictions[i] < 0.5:
            predict_right += 1

    P = predict_right / max(predict, 0.000001)
    R = predict_right / totel_right
    F = 2 * P * R / max((P + R), 0.000001)
    print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)

    print('test distance > 0.5  = ', predict_right05 / totel_right)

    return P, R, F


def test_rank(nn_model, tagDict_test):

    word_vob, word_id2word, target_vob, target_id2word, max_s, \
    target_vob_train = ProcessData_Siamese.get_word_index([trainfile])

    KGem_RankDict = ProcessData_Siamese.get_relembed_sim_rank(target_vob_train, type_W)

    pairs_test0, labels_test = ProcessData_Siamese.CreatePairs(tagDict_test, istest=False)
    print('CreatePairs test len = ', len(pairs_test0[0]), len(pairs_test0))

    test_x1_sent = np.asarray(pairs_test0[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test0[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test0[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test0[3], dtype="int32")
    test_x1_sent_cahr = np.asarray(pairs_test0[4], dtype="int32")
    test_y = np.asarray(labels_test, dtype="int32")

    inputs_test_x = [test_x1_sent, test_x1_e1_posi, test_x1_e2_posi, test_x2_tag, test_x1_sent_cahr]
    inputs_test_y = [test_y]

    loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, batch_size=batch_size, verbose=0)
    print('test in test_model--- loss, acc', loss, acc)

    data_s_list, data_tag_list, data_e1_posi_list, data_e2_posi_list, char_s_list = pairs_test0

    P = 0
    R = 0
    F = 0
    predict = 0
    predict_right = 0

    totel_right = len(data_s_list) / 2
    # totel_right = len(data_s_list[:1000]) /2

    labels_all = []
    data_s_all = []
    data_e1_posi_all = []
    data_e2_posi_all = []
    data_tag_all = []
    char_s_all = []

    truth_tag_list = []

    for i in range(len(data_s_list)):
        if labels_test[i] == 0:
            continue
        # print(i)

        for ins in target_vob_train.values():

            data_s_all.append(data_s_list[i])
            data_tag_all.append([ins])
            data_e1_posi_all.append(data_e1_posi_list[i])
            data_e2_posi_all.append(data_e2_posi_list[i])
            truth_tag_list.append(data_tag_list[i][0])
            char_s_all.append(char_s_list[i])

    pairs_test = [data_s_all, data_tag_all, data_e1_posi_all, data_e2_posi_all, char_s_all]

    test_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test[3], dtype="int32")
    test_x1_sent_cahr = np.asarray(pairs_test[4], dtype="int32")

    predictions = nn_model.predict([test_x1_sent, test_x1_e1_posi, test_x1_e2_posi,
                                    test_x2_tag, test_x1_sent_cahr], batch_size=batch_size, verbose=0)

    if len(predictions) > 2 and len(predictions[0]) == 1:
        print('-.- -.- -.- -.- -.- -.- -.- -.- -.- len(predictions) > 2 and len(predictions[0]) == 1', len(predictions))
        target_vob_train_len = len(target_vob_train)
        assert len(predictions) // target_vob_train_len == totel_right
        assert len(truth_tag_list) == totel_right

        for i in range(len(predictions) // target_vob_train_len):
            left = i * target_vob_train_len
            right = (i + 1) * target_vob_train_len
            subpredictions = predictions[left:right]
            subpredictions = subpredictions.flatten().tolist()

            distantDict = {}
            for j, disvlaue in enumerate(subpredictions):
                distantDict[data_tag_all[i+j][0]] = disvlaue

            distantList = sorted(distantDict.items(), key=lambda s: s[1], reverse=True)
            distantDict = dict(distantList)
            distantList = list(distantDict.keys())

            match_max = -1
            match_max_where = -1

            for rk in KGem_RankDict.keys():
                rlist = KGem_RankDict[rk]

                match = 0

                for rli in range(len(rlist)):
                    if rlist[rli] == distantList[rli]:
                        match += 1
                    else:
                        break

                if match > match_max:
                    match_max_where = rk
                    match_max = match

            if match_max > 1:
                predict += 1

                if match_max_where == truth_tag_list[i]:
                    predict_right += 1

        P = predict_right / max(predict, 0.000001)
        R = predict_right / totel_right
        F = 2 * P * R / max((P + R), 0.000001)
        print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)

    return P, R, F


def test_model(nn_model, tagDict_test, target_vob):

    pairs_test0, labels_test = ProcessData_Siamese.CreatePairs(tagDict_test, istest=False)
    print('CreatePairs test len = ', len(pairs_test0[0]), len(pairs_test0))

    test_x1_sent = np.asarray(pairs_test0[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test0[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test0[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test0[3], dtype="int32")
    test_x1_sent_cahr = np.asarray(pairs_test0[4], dtype="int32")
    test_y = np.asarray(labels_test, dtype="int32")

    inputs_test_x = [test_x1_sent, test_x1_e1_posi, test_x1_e2_posi, test_x2_tag, test_x1_sent_cahr]
    inputs_test_y = [test_y]

    loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, batch_size=batch_size, verbose=0)
    print('test in test_model--- loss, acc', loss, acc)

    data_s_list, data_tag_list, data_e1_posi_list, data_e2_posi_list, char_s_list = pairs_test0

    P = 0
    R = 0
    F = 0
    predict = 0
    predict_right = 0
    predict_right_c = 0
    predict_c = 0
    predict_right05 = 0
    totel_right = len(data_s_list) / 2
    # totel_right = len(data_s_list[:1000]) /2

    labels_all = []
    data_s_all = []
    data_e1_posi_all = []
    data_e2_posi_all = []
    data_tag_all = []
    char_s_all = []

    truth_tag_list = []

    for i in range(len(data_s_list)):
        if labels_test[i] == 0:
            continue
        # print(i)
        for numi, ins in enumerate(target_vob.values()):
            assert numi == ins

            data_s_all.append(data_s_list[i])
            data_tag_all.append([ins])
            data_e1_posi_all.append(data_e1_posi_list[i])
            data_e2_posi_all.append(data_e2_posi_list[i])
            if data_tag_list[i][0] == ins:
                labels_all.append(1)
                truth_tag_list.append([ins])
                # print(ins)
            else:
                labels_all.append(0)
            char_s_all.append(char_s_list[i])

    pairs_test = [data_s_all, data_tag_all, data_e1_posi_all, data_e2_posi_all, char_s_all]

    test_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test[3], dtype="int32")
    test_x1_sent_cahr = np.asarray(pairs_test[4], dtype="int32")
    test_y = np.asarray(labels_all, dtype="int32")

    predictions = nn_model.predict([test_x1_sent, test_x1_e1_posi, test_x1_e2_posi,
                                    test_x2_tag, test_x1_sent_cahr], batch_size=len(target_vob), verbose=0)

    if len(predictions) > 2 and len(predictions[0]) == 1:
        print('-.- -.- -.- -.- -.- -.- -.- -.- -.- len(predictions) > 2 and len(predictions[0]) == 1', len(predictions))
        assert len(predictions) // len(target_vob) == totel_right
        assert len(truth_tag_list) == totel_right
        predict_rank = 0

        for i in range(len(predictions)//len(target_vob)):
            left = i*len(target_vob)
            right = (i+1)*len(target_vob)
            subpredictions = predictions[left:right]
            subpredictions = subpredictions.flatten().tolist()

            distantDict = {}
            for num, disvlaue in enumerate(subpredictions):
                distantDict[num] = disvlaue

            # for dd in distantDict:
            #     print('!!!', dd)

            distantList = sorted(distantDict.items(), key=lambda s: s[1], reverse=True)
            distantDict = dict(distantList)
            distantList = list(distantDict.keys())
            target_where = distantList.index(truth_tag_list[i][0]) + 1

            predict_rank += target_where


            mindis = max(subpredictions)
            mindis_where = subpredictions.index(mindis)

            if mindis > 0.5:
                predict += 1

                if mindis_where == truth_tag_list[i][0]:
                    predict_right += 1


            if subpredictions[truth_tag_list[i][0]] > 0.5:
                predict_right05 += 1


        P = predict_right / max(predict, 0.000001)
        R = predict_right / totel_right
        F = 2 * P * R / max((P + R), 0.000001)
        print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
        print('test predict_rank = ', predict_rank / totel_right)
        print('test distance > 0.5  = ', predict_right05 / totel_right)

    return P, R, F


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                    resultdir,
                    npoches=100, batch_size=50, retrain=False, inum=0):

    if retrain:
        nn_model.load_weights(modelfile)
        modelfile = modelfile + '.2nd.h5'

    nn_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    checkpointer = ModelCheckpoint(filepath=modelfile + ".best_model.h5", monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001)

    # nn_model.fit(inputs_train_x, inputs_train_y,
    #              batch_size=batch_size,
    #              epochs=npoches,
    #              verbose=1,
    #              shuffle=True,
    #              validation_split=0.1,
    #
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # nn_model.save_weights(modelfile, overwrite=True)
    #
    # print('the test result-----------------------')
    # P, R, F = test_model(nn_model, pairs_test, labels_test, classifer_labels_test, target_vob)
    # print('P = ', P, 'R = ', R, 'F = ', F)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1

        inputs_train_x, inputs_train_y = Dynamic_get_trainSet(tagDict_train)
        inputs_dev_x, inputs_dev_y = Dynamic_get_trainSet(tagDict_dev)

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_data=[inputs_dev_x, inputs_dev_y],
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
                               verbose=1,
                               callbacks=[reduce_lr, checkpointer])

        print('the test result-----------------------')

        P, R, F = test_model(nn_model, tagDict_test, target_vob)

        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(str(inum), nowepoch, P, R, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 50:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, resultdir):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    print('the test result-----------------------')
    P, R, F = test_model(nnmodel, tagDict_test, target_vob)
    print('P = ', P, 'R = ', R, 'F = ', F)

    print('the test rank result-----------------------')
    P, R, F = test_rank(nnmodel, tagDict_test)
    print('P = ', P, 'R = ', R, 'F = ', F)

    # print('the test_model_4trainset result-----------------------')
    # P, R, F = test_model_4trainset(nnmodel, pairs_train, labels_train, classifer_labels_train, target_vob)
    # print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, wordvocabsize, tagvocabsize, posivocabsize,charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,
                     batch_size=32):
    nn_model = None

    if modelname is 'Model_BiLSTM_sent_MLP__KGembed':
        nn_model = Model_BiLSTM_sent_MLP__KGembed(wordvocabsize=wordvocabsize,
                                                  tagvocabsize=tagvocabsize,
                                                  posivocabsize=posivocabsize,
                                                  charvocabsize=charvocabsize,
                                                  word_W=word_W, posi_W=posi_W, tag_W=tag_W, char_W=char_W,
                                                  input_sent_lenth=input_sent_lenth,
                                                  input_maxword_length=max_c,
                                                  w2v_k=w2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k,c2v_k=c2v_k,
                                                  batch_size=batch_size)

    if modelname is 'Model_BiLSTM_sent_linear__KGembed':
        nn_model = Model_BiLSTM_sent_linear__KGembed(wordvocabsize=wordvocabsize,
                                                  tagvocabsize=tagvocabsize,
                                                  posivocabsize=posivocabsize,
                                                  charvocabsize=charvocabsize,
                                                  word_W=word_W, posi_W=posi_W, tag_W=tag_W, char_W=char_W,
                                                  input_sent_lenth=input_sent_lenth,
                                                  input_maxword_length=max_c,
                                                  w2v_k=w2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k, c2v_k=c2v_k,
                                                  batch_size=batch_size)

    return nn_model

def Dynamic_get_trainSet(tagDict):

    pairs_train, labels_train = ProcessData_Siamese.CreatePairs(tagDict, istest=False)

    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x2_tag = np.asarray(pairs_train[1], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs_train[2], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs_train[3], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs_train[4], dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")
    # train_y_classifer = np.asarray(classifer_labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x2_tag, train_x1_sent_cahr]
    inputs_train_y = [train_y]

    return inputs_train_x, inputs_train_y


if __name__ == "__main__":

    maxlen = 50

    modelname = 'Model_BiLSTM_sent_linear__KGembed'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"

    t2v_file = './data/KG2v/FB15K_PTransE_Relation2Vec_100.txt'
    trainfile = './data/annotated_fb__zeroshot_RE.random.train.txt'
    testfile = './data/annotated_fb__zeroshot_RE.random.test.txt'

    # t2v_file = './data/FewRel/FewRel.rel2v.by_glove.100d.txt'
    # trainfile = './data/FewRel/FewRel_data.train.txt'
    # testfile = './data/FewRel/FewRel_data.test.txt'

    resultdir = "./data/result/"

    datafname = 'data_Siamese.split.WordChar.PTransE'
    # datafname = 'FewRel_data_Siamese.split.WordChar.rel2v_by_glove'

    datafile = "./model/model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 256 #16,

    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_Siamese.get_data(trainfile, testfile, w2v_file, c2v_file, t2v_file, datafile,
                 w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen, hasNeg=hasNeg, percent=0.05)


    tagDict_train, tagDict_dev, tagDict_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    char_vob, char_id2char, char_W, c2v_k,\
    target_vob, target_id2word, type_W, type_k,\
    posi_W, posi_k,\
    max_s, max_posi, max_c = pickle.load(open(datafile, 'rb'))


    nn_model = SelectModel(modelname,
                           wordvocabsize=len(word_vob),
                           tagvocabsize=len(target_vob),
                           posivocabsize=max_posi+1,
                           charvocabsize=len(char_vob),
                           word_W=word_W, posi_W=posi_W, tag_W=type_W, char_W=char_W,
                           input_sent_lenth=max_s,
                           w2v_k=w2v_k, posi2v_k=max_posi+1, tag2v_k=type_k, c2v_k=c2v_k,
                           batch_size=batch_size)

    for inum in range(1, 3):

        modelfile = "./model/" + modelname + "__" + datafname + "__" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x=[], inputs_train_y=[],
                            resultdir=resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x=[], inputs_train_y=[],
                                resultdir=resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, modelname, modelfile, resultdir)


# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

