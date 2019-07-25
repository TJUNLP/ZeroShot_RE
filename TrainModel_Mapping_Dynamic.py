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
import ProcessData_Mapping
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_Mapping import Model_sent_MLP__KGembed



def test_model(nn_model, tag2sentDict_test):

    predict = 0
    predict_right = 0
    predict_right05 = 0

    data_s_all = []
    data_tag_all = []
    labels_all = []
    totel_right = 0
    truth_tag_list = []
    for tag in tag2sentDict_test.keys():
        for sent in tag2sentDict_test[tag]:
            totel_right += 1

            for numi, ins in enumerate(target_vob.values()):
                assert numi == ins

                data_s_all.append(sent)
                data_tag_all.append([ins])

                if tag == ins:
                    labels_all.append(1)
                    truth_tag_list.append(tag)
                else:
                    labels_all.append(0)

    pairs_test = [data_s_all, data_tag_all]

    test_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test[1], dtype="int32")

    inputs_train_x = [test_x1_sent, test_x2_tag]

    predictions = nn_model.predict(inputs_train_x, batch_size=batch_size, verbose=0)

    assert len(predictions) // len(target_vob) == totel_right
    assert len(truth_tag_list) == totel_right
    predict_rank = 0

    for i in range(len(predictions) // len(target_vob)):
        left = i * len(target_vob)
        right = (i + 1) * len(target_vob)
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

    return P, R, F


def Dynamic_get_trainSet():
    pairs_train, labels_train = ProcessData_Mapping.CreatePairs(tag2sentDict_train)

    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x2_tag = np.asarray(pairs_train[1], dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x2_tag]
    inputs_train_y = [train_y]

    return inputs_train_x, inputs_train_y


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

        inputs_train_x, inputs_train_y = Dynamic_get_trainSet()

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_split=0.2,
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
                               verbose=1,
                               callbacks=[reduce_lr, checkpointer])

        print('the test result-----------------------')
        P, R, F = test_model(nn_model, tag2sentDict_test)

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
    P, R, F = test_model(nnmodel, tag2sentDict_test)
    print('P = ', P, 'R = ', R, 'F = ', F)

    # print('the test_model_4trainset result-----------------------')
    # P, R, F = test_model_4trainset(nnmodel, pairs_train, labels_train, classifer_labels_train, target_vob)
    # print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, sentvocabsize, tagvocabsize,
                sent_W, tag_W, s2v_k, tag2v_k):

    nn_model = None

    if modelname is 'Model_sent_MLP__KGembed':
        nn_model = Model_sent_MLP__KGembed(sentvocabsize=sentvocabsize,
                                                  tagvocabsize=tagvocabsize,
                                                  sent_W=sent_W, tag_W=tag_W,
                                                  s2v_k=s2v_k, tag2v_k=tag2v_k)

    return nn_model


if __name__ == "__main__":

    maxlen = 50

    modelname = 'Model_sent_MLP__KGembed'

    print(modelname)

    t2v_file = './data/KG2v/FB15K_PTransE_Relation2Vec_100.txt'
    s2v_trainfile = './data/Model_BiLSTM_SentPair_1__data_Siamese.WordChar.Sentpair__1.h5.train.txt'
    s2v_testfile = './data/Model_BiLSTM_SentPair_1__data_Siamese.WordChar.Sentpair__1.h5.test.txt'
    resultdir = "./data/result/"

    # datafname = 'data_Siamese.4_allneg' #1,3, 4_allneg, 4_allneg_segmentNeg
    datafname = 'data_Mapping.PTransE'

    datafile = "./model/model_data/" + datafname + ".pkl"

    sentpair_datafile = "./model/model_data/data_Siamese.WordChar.Sentpair.pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 256

    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_Mapping.get_data(sentpair_datafile,
                                     s2v_trainfile, s2v_testfile, t2v_file, datafile, s2v_k=400, t2v_k=100)

    tag2sentDict_train, tag2sentDict_test,\
    sent_W, sent_k, \
    target_vob, target_id2word, type_W, type_k = pickle.load(open(datafile, 'rb'))

    nn_model = SelectModel(modelname,
                           sentvocabsize=len(sent_W), tagvocabsize=len(target_vob),
                           sent_W=sent_W, tag_W=type_W, s2v_k=sent_k, tag2v_k=type_k)

    for inum in range(1, 3):

        modelfile = "./model/" + modelname + "__" + datafname + "__" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("data has extisted: " + datafile)
            print("Training model....")
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

