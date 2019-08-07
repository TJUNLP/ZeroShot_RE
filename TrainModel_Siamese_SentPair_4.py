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
import ProcessData_Siamese_SentPair
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_Siamese_SentPair import Model_BiLSTM_SentPair_1
from NNstruc.NN_Siamese_SentPair import Model_BiLSTM_SentPair_3
from NNstruc.NN_Siamese_SentPair import Model_BiLSTM_SentPair_RelPunish_3
import keras


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


def get_sent_index(nn_model, inputs_train_x, tagIndex, w2file):

    sent_vob = {}

    # intermediate_layer_model_x1 = keras.models.Model(inputs=nn_model.input,
    #                                               outputs=nn_model.get_layer('bidirectional_1').get_output_at(0))
    # intermediate_output_x1 = intermediate_layer_model_x1.predict(inputs_train_x)

    intermediate_layer_model_x2 = keras.models.Model(inputs=nn_model.input,
                                                  outputs=nn_model.get_layer('bidirectional_1').get_output_at(1))
    intermediate_output_x2 = intermediate_layer_model_x2.predict(inputs_train_x)

    fw = codecs.open(w2file, 'w', encoding='utf-8')
    inx = 0
    for i, op in enumerate(intermediate_output_x2):

        key = (inx, tagIndex[i])
        sent_vob[key] = op
        inx += 1
        fw.write(str(key[0]) + ' ' + str(key[1]))
        for item in op:
            fw.write(' ' + str(item))
        fw.write('\n')
        # key = (inx, tagIndex[1])
        # sent_vob[key] = intermediate_output_x2[i]
        # print(i, len(sent_vob[key]))
        # inx += 1
    print(inx, len(sent_vob))



def test_model2(nn_model, tag2sentDict_test):

    predict = 0
    predict_right = 0
    predict_right05 = 0


    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_tag_all = []

    labels_all = []
    totel_right = 0

    truth_tag_list = []
    for tag in tag2sentDict_test.keys():
        sents = tag2sentDict_test[tag]

        for s in range(1, len(sents)):
            totel_right += 1

            for si, ty in enumerate(tag2sentDict_test.keys()):

                data_s, data_e1_posi, data_e2_posi, char_s = tag2sentDict_test[ty][0]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)
                data_tag_all.append([ty])

                data_s, data_e1_posi, data_e2_posi, char_s = sents[s]
                data_s_all_1.append(data_s)
                data_e1_posi_all_1.append(data_e1_posi)
                data_e2_posi_all_1.append(data_e2_posi)
                char_s_all_1.append(char_s)

                if tag == ty:
                    labels_all.append(1)
                    truth_tag_list.append(si)
                else:
                    labels_all.append(0)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1, data_tag_all]

    train_x1_sent = np.asarray(pairs[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs[3], dtype="int32")
    train_x2_sent = np.asarray(pairs[4], dtype="int32")
    train_x2_e1_posi = np.asarray(pairs[5], dtype="int32")
    train_x2_e2_posi = np.asarray(pairs[6], dtype="int32")
    train_x2_sent_cahr = np.asarray(pairs[7], dtype="int32")
    train_tag = np.asarray(pairs[8], dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_x2_sent, train_x2_e1_posi, train_x2_e2_posi, train_x2_sent_cahr, train_tag]

    predictions = nn_model.predict(inputs_train_x, batch_size=batch_size, verbose=1)

    if len(predictions) < 10:
        predictions = predictions[0]

    width = len(tag2sentDict_test.keys())
    assert len(predictions) // width == totel_right
    assert len(truth_tag_list) == totel_right
    predict_rank = 0

    for i in range(len(predictions) // width) :
        left = i * width
        right = (i + 1) * width
        subpredictions = predictions[left:right]
        subpredictions = subpredictions.flatten().tolist()

        distantDict = {}
        for num, disvlaue in enumerate(subpredictions):
            distantDict[num] = disvlaue

        distantList = sorted(distantDict.items(), key=lambda s: s[1], reverse=True)
        distantDict = dict(distantList)
        distantList = list(distantDict.keys())
        target_where = distantList.index(truth_tag_list[i]) + 1
        predict_rank += target_where

        mindis = max(subpredictions)
        mindis_where = subpredictions.index(mindis)

        if mindis > 0.5:
            predict += 1

            if mindis_where == truth_tag_list[i]:
                predict_right += 1

        if subpredictions[truth_tag_list[i]] > 0.5:
            predict_right05 += 1

    P = predict_right / max(predict, 0.000001)
    R = predict_right / totel_right
    F = 2 * P * R / max((P + R), 0.000001)
    print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
    print('test predict_rank = ', predict_rank / totel_right)
    print('test distance > 0.5  = ', predict_right05 / totel_right)
    print('P =, R =, F = ', P, R, F)
    return P, R, F


def test_model(nn_model, tagDict_test, needembed=False, w2file=''):

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_tag_all = []

    tagIndex = []

    for tag in tagDict_test.keys():

        if needembed == True:
            star = 0
        else:
            star = 1
            if len(tagDict_test[tag]) < 2:
                continue

        for i in range(star, len(tagDict_test[tag])):

            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_test[tag][0]
            data_s_all_0.append(data_s)
            data_e1_posi_all_0.append(data_e1_posi)
            data_e2_posi_all_0.append(data_e2_posi)
            char_s_all_0.append(char_s)

            data_tag_all.append([tag])

            data_s, data_e1_posi, data_e2_posi, char_s = tagDict_test[tag][i]
            data_s_all_1.append(data_s)
            data_e1_posi_all_1.append(data_e1_posi)
            data_e2_posi_all_1.append(data_e2_posi)
            char_s_all_1.append(char_s)
            tagIndex.append(tag)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1, data_tag_all]

    train_x1_sent = np.asarray(pairs[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs[3], dtype="int32")
    train_x2_sent = np.asarray(pairs[4], dtype="int32")
    train_x2_e1_posi = np.asarray(pairs[5], dtype="int32")
    train_x2_e2_posi = np.asarray(pairs[6], dtype="int32")
    train_x2_sent_cahr = np.asarray(pairs[7], dtype="int32")
    train_tag = np.asarray(pairs[8], dtype="int32")
    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_x2_sent, train_x2_e1_posi, train_x2_e2_posi, train_x2_sent_cahr, train_tag]

    predict = 0
    predict_right = 0
    totel_right = len(pairs[0])

    predictions = nn_model.predict(inputs_train_x, batch_size=batch_size, verbose=0)

    if len(predictions) < 10:
        predictions = predictions[0]

    if needembed == True:
        get_sent_index(nn_model, inputs_train_x, tagIndex, w2file)

    for pre in predictions:

        if pre > 0.5:
            predict += 1
            predict_right += 1

    P = predict_right / predict
    R = predict_right / totel_right
    F = 2 * P * R / (P + R)
    print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
    print('P =, R =, F = ', P, R, F)

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

        inputs_train_x, inputs_train_y = Dynamic_get_trainSet(istest=False)
        inputs_dev_x, inputs_dev_y = Dynamic_get_trainSet(istest=True)

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_data=[inputs_dev_x, inputs_dev_y],
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
                               verbose=1,
                               callbacks=[reduce_lr, checkpointer])

        print('the test result-----------------------')
        # loss, acc = nn_model.evaluate(inputs_dev_x, inputs_dev_y, batch_size=batch_size, verbose=0)
        P, R, F = test_model(nn_model, tagDict_test, needembed=False)
        P, R, F = test_model2(nn_model, tagDict_test)
        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(str(inum), nowepoch, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 15:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, resultdir, w2file=''):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    print('the test result-----------------------')
    P, R, F = test_model(nn_model, tagDict_test, needembed=False)
    print('P = ', P, 'R = ', R, 'F = ', F)
    print('the test 2 result-----------------------')
    P, R, F = test_model2(nn_model, tagDict_test)
    print('P = ', P, 'R = ', R, 'F = ', F)

    # print('the train sent representation-----------------------')
    # P, R, F = test_model(nn_model, tagDict_train, needembed=True, w2file=w2file+'.train.txt')
    # print('P = ', P, 'R = ', R, 'F = ', F)
    #
    # print('the test sent representation-----------------------')
    # P, R, F = test_model(nn_model, tagDict_test, needembed=True, w2file=w2file+'.test.txt')
    # print('P = ', P, 'R = ', R, 'F = ', F)

    # print('the test_model_4trainset result-----------------------')
    # P, R, F = test_model_4trainset(nnmodel, pairs_train, labels_train, classifer_labels_train, target_vob)
    # print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, wordvocabsize, tagvocabsize, posivocabsize,charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,
                     batch_size=32):
    nn_model = None

    if modelname is 'Model_BiLSTM_SentPair_RelPunish_3':
        nn_model = Model_BiLSTM_SentPair_RelPunish_3(wordvocabsize=wordvocabsize,
                                                  posivocabsize=posivocabsize,
                                                  charvocabsize=charvocabsize,
                                                    tagvocabsize=tagvocabsize,
                                                  word_W=word_W, posi_W=posi_W, char_W=char_W, tag_W=tag_W,
                                                  input_sent_lenth=input_sent_lenth,
                                                  input_maxword_length=max_c,
                                                  w2v_k=w2v_k, posi2v_k=posi2v_k, c2v_k=c2v_k, tag2v_k=tag2v_k,
                                                  batch_size=batch_size)

    if modelname is 'Model_BiLSTM_SentPair_RelPunish_3_05':

        print('!!!!!!!!!!!!!!')
        nn_model = Model_BiLSTM_SentPair_RelPunish_3(wordvocabsize=wordvocabsize,
                                                  posivocabsize=posivocabsize,
                                                  charvocabsize=charvocabsize,
                                                    tagvocabsize=tagvocabsize,
                                                  word_W=word_W, posi_W=posi_W, char_W=char_W, tag_W=tag_W,
                                                  input_sent_lenth=input_sent_lenth,
                                                  input_maxword_length=max_c,
                                                  w2v_k=w2v_k, posi2v_k=posi2v_k, c2v_k=c2v_k, tag2v_k=tag2v_k,
                                                  batch_size=batch_size)


    return nn_model


def Dynamic_get_trainSet(istest):

    if istest == True:
        tagDict = tagDict_dev
    else:
        tagDict = tagDict_train

    pairs_train, labels_train = ProcessData_Siamese_SentPair.CreatePairs(tagDict, istest=istest)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))


    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs_train[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs_train[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs_train[3], dtype="int32")
    train_x2_sent = np.asarray(pairs_train[4], dtype="int32")
    train_x2_e1_posi = np.asarray(pairs_train[5], dtype="int32")
    train_x2_e2_posi = np.asarray(pairs_train[6], dtype="int32")
    train_x2_sent_cahr = np.asarray(pairs_train[7], dtype="int32")
    train_tag = np.asarray(pairs_train[8], dtype="int32")

    train_y = np.asarray(labels_train, dtype="int32")
    train_p = np.ones(len(labels_train), dtype='float32')
    # train_y_classifer = np.asarray(classifer_labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_x2_sent, train_x2_e1_posi, train_x2_e2_posi, train_x2_sent_cahr, train_tag]
    inputs_train_y = [train_y, train_p]

    return inputs_train_x, inputs_train_y


if __name__ == "__main__":

    maxlen = 100

    # modelname = 'Model_BiLSTM_SentPair_2'
    # modelname = 'Model_BiLSTM_SentPair_3'
    modelname = 'Model_BiLSTM_SentPair_RelPunish_3_05'
    modelname = 'Model_BiLSTM_SentPair_RelPunish_3'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    t2v_file = './data/WikiReading/WikiReading.rel2v.by_glove.100d.txt'

    # trainfile = './data/annotated_fb__zeroshot_RE.random.train.txt'
    # testfile = './data/annotated_fb__zeroshot_RE.random.test.txt'

    # trainfile = './data/FewRel/FewRel_data.train.txt'
    # testfile = './data/FewRel/FewRel_data.test.txt'

    trainfile = './data/WikiReading/WikiReading_data.random.train.txt'
    testfile = './data/WikiReading/WikiReading_data.random.test.txt'

    resultdir = "./data/result/"

    # datafname = 'FewRel_data_Siamese.WordChar.Sentpair'
    datafname = 'WikiReading_data_Siamese.WordChar.Sentpair.relPunish'

    datafile = "./model/model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 512 #256

    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_Siamese_SentPair.get_data(trainfile, testfile, w2v_file, c2v_file, t2v_file, datafile,
                 w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen, hasNeg=hasNeg, percent=0.05)

    tagDict_train, tagDict_dev, tagDict_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    char_vob, char_id2char, char_W, c2v_k,\
    target_vob, target_id2word,\
    posi_W, posi_k, type_W, type_k,\
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

    for inum in range(0, 2):

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
            infer_e2e_model(nn_model, modelname, modelfile, resultdir,w2file=modelfile)


# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

