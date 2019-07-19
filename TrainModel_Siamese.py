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
from ProcessData_Siamese import get_data
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_Siamese import Model_BiLSTM_sent__MLP_KGembed


def test_model(nn_model, pairs_test0, labels_test, classifer_labels_test, target_vob):

    data_s_list, data_tag_list, data_e1_posi_list, data_e2_posi_list = pairs_test0

    predict = 0
    predict_right = 0
    predict_right_c = 0
    predict_c = 0
    totel_right = min(50, len(data_s_list))

    labels_all = []
    data_s_all = []
    data_e1_posi_all = []
    data_e2_posi_all = []
    data_tag_all = []

    for i in range(min(50, len(data_s_list))):
        # print(i)
        for ins in target_vob.values():

            data_s_all.append(data_s_list[i])
            data_tag_all.append([ins])
            data_e1_posi_all.append(data_e1_posi_list[i])
            data_e2_posi_all.append(data_e2_posi_list[i])
            if data_tag_list[i][0] == ins:
                labels_all.append(1)
                # print(ins)
            else:
                labels_all.append(0)

    pairs_test = [data_s_all, data_tag_all, data_e1_posi_all, data_e2_posi_all]

    test_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    test_x2_tag = np.asarray(pairs_test[1], dtype="int32")
    test_x1_e1_posi = np.asarray(pairs_test[2], dtype="int32")
    test_x1_e2_posi = np.asarray(pairs_test[3], dtype="int32")
    test_y = np.asarray(labels_all, dtype="int32")

    predictions = nn_model.predict([test_x1_sent, test_x1_e1_posi, test_x1_e2_posi,
                                    test_x2_tag], batch_size=512, verbose=0)

    if len(predictions) > 2 and len(predictions[0]) == 1:
        print('-.- -.- -.- -.- -.- -.- -.- -.- -.- len(predictions) > 2 and len(predictions[0]) == 1')
        assert len(predictions) // len(target_vob) == totel_right
        for i in range(len(predictions)//len(target_vob)):
            subpredictions = predictions[i*len(target_vob):i*len(target_vob) + len(target_vob)]
            subpredictions = subpredictions.flatten().tolist()

            mindis = min(subpredictions)
            mindis_where = subpredictions.index(min(subpredictions))

            # mincount = 0
            # for num, disvlaue in enumerate(subpredictions):
            #     if disvlaue < 0.5:
            #         mindis_where = num
            #         mincount += 1
            # if mincount == 1:
            #     predict += 1
            #     if mindis_where == fragment_tag_list[i]:
            #         predict_right += 1

            if mindis < 0.5:
                predict += 1

                if mindis_where == data_tag_list[i][0]:
                    predict_right += 1



        P = predict_right / max(predict, 0.000001)
        R = predict_right / totel_right
        F = 2 * P * R / max((P + R), 0.000001)
        print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)


    elif len(predictions) > 2 and len(predictions[0]) == 2:

        print('-.- -.- -.- -.- -.- -.- -.- -.- -.- len(predictions) > 2 and len(predictions[0]) == 2')
        assert len(predictions) // len(target_vob) == len(fragment_tag_list)
        for i in range(len(predictions) // len(target_vob)):
            subpredictions = predictions[i * len(target_vob):i * len(target_vob) + len(target_vob)]

            max_1 = 0
            max_where = -1
            for num, ptagindex in enumerate(subpredictions):
                # print(num, ptagindex)
                if max_1 < ptagindex[1]:
                    max_1 = ptagindex[1]
                    max_where = num

            ptag = candidate_tag_list[i][max_where]

            ttag = fragment_tag_list[i]
            # print(max_where, candidate_tag_list[i], 'ptag', ptag, 'ttag', ttag)

            predict_c += 1
            if ptag == ttag:
                predict_right_c += 1

        P = predict_right_c / predict_c
        R = predict_right_c / totel_right
        F = 2 * P * R / (P + R)
        print('BiClassifer!!!!!!!!!! predict_right =, predict =, target =, ', predict_right_c, predict_c, totel_right)
        print('Biclassifer!!!!!!!!!! P= ', P, 'R= ', R, 'F= ', F)


    elif len(predictions) > 2 and len(predictions[0]) == len(target_vob):

        print('-.- -.- -.- -.- -.- -.- -.- -.- -.- len(predictions) > 2 and len(predictions[0]) == 4')
        assert len(predictions) // len(target_vob) == len(fragment_tag_list)
        for i in range(len(predictions) // len(target_vob)):
            subpredictions = predictions[i * len(target_vob):i * len(target_vob) + len(target_vob)]

            ptag_npall = np.zeros(len(target_vob), dtype='float32')
            for num, ptagindex in enumerate(subpredictions):
                ptag_npall += ptagindex

            ptag = np.argmax(ptag_npall)

            if ptag != 'NULL':
                predict_c += 1

            ttag = fragment_tag_list[i]

            if ptag == ttag and ttag != 'NULL':
                predict_right_c += 1

        P = predict_right_c / predict_c
        R = predict_right_c / totel_right
        F = 2 * P * R / (P + R)
        print('Classifer!!!!!!!!!! predict_right =, predict =, target =, ', predict_right_c, predict_c, totel_right)
        print('classifer!!!!!!!!!! P= ', P, 'R= ', R, 'F= ', F)


    else:
        assert len(predictions[1]) // len(target_vob) == len(fragment_tag_list)
        for i in range(len(predictions[1]) // len(target_vob)):
            subpredictions = predictions[1][i * len(target_vob):i * len(target_vob) + len(target_vob)]
            subpredictions = subpredictions.flatten().tolist()

            mindis = min(subpredictions)
            mindis_where = subpredictions.index(min(subpredictions))

            # mincount = 0
            # for num, disvlaue in enumerate(subpredictions):
            #     if disvlaue < 0.5:
            #         mindis_where = num
            #         mincount += 1
            # if mincount == 1:
            #     predict += 1
            #     if mindis_where == fragment_tag_list[i]:
            #         predict_right += 1

            if mindis < 0.5:
                predict += 1
                if mindis_where == fragment_tag_list[i]:
                    predict_right += 1

            subpredictions = predictions[0][i * len(target_vob):i * len(target_vob) + len(target_vob)]
            ptag_npall = np.zeros(len(target_vob),dtype='float32')
            for num, ptagindex in enumerate(subpredictions):
                ptag_npall += ptagindex

            ptag = np.argmax(ptag_npall)

            if ptag != 'NULL':
                predict_c += 1

            ttag = fragment_tag_list[i]

            if ptag == ttag and ttag != 'NULL':
                predict_right_c += 1

        P = predict_right / predict
        R = predict_right / totel_right
        F = 2 * P * R / (P + R)
        print('Distance!!!!!!!!!! predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
        print('Distance!!!!!!!!!! P= ', P, 'R= ', R, 'F= ', F)
        P = predict_right_c / predict_c
        R = predict_right_c / totel_right
        F = 2 * P * R / (P + R)
        print('Classifer!!!!!!!!!! predict_right =, predict =, target =, ', predict_right_c, predict_c, totel_right)
        print('classifer!!!!!!!!!! P= ', P, 'R= ', R, 'F= ', F)

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

    # nn_model.fit(x_word, y,
    #              batch_size=batch_size,
    #              epochs=npochos,
    #              verbose=1,
    #              shuffle=True,
    #              # validation_split=0.2,
    #              validation_data=(x_word_val, y_val),
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # save_model(nn_model, modelfile)
    # # nn_model.save(modelfile, overwrite=True)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_split=0.2,
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
                               verbose=1,
                               callbacks=[reduce_lr, checkpointer])

        print('the test result-----------------------')
        P, R, F = test_model(nn_model, pairs_test, labels_test, classifer_labels_test, target_vob)

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
    P, R, F = test_model(nnmodel, pairs_test, labels_test, classifer_labels_test, target_vob)
    print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, wordvocabsize, tagvocabsize, posivocabsize,charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     input_sent_lenth,
                     w2v_k, posi2v_k, tag2v_k, c2v_k, tag_k,
                     batch_size=32):
    nn_model = None

    if modelname is 'Model_BiLSTM_sent__MLP_KGembed':
        nn_model = Model_BiLSTM_sent__MLP_KGembed(wordvocabsize=wordvocabsize,
                                     tagvocabsize=tagvocabsize,
                                     posivocabsize=posivocabsize,
                                     word_W=word_W, posi_W=posi_W, tag_W=tag_W,
                                     input_sent_lenth=input_sent_lenth,
                                     w2v_k=w2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k,
                                     batch_size=batch_size)

    return nn_model


if __name__ == "__main__":

    maxlen = 50

    modelname = 'Model_BiLSTM_sent__MLP_KGembed'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    t2v_file = './data/KG2v/FB15K_OpenKETransE_Relation2Vec_100.txt'
    trainfile = './data/annotated_fb__zeroshot_RE.random.train.txt'
    testfile = './data/annotated_fb__zeroshot_RE.random.test.txt'
    resultdir = "./data/result/"

    # datafname = 'data_Siamese.4_allneg' #1,3, 4_allneg, 4_allneg_segmentNeg
    datafname = 'data_Siamese.4.onlyword.onlyKGem'

    datafile = "./model/model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 256 #16,

    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        get_data(trainfile, testfile, w2v_file, c2v_file, t2v_file, datafile,
                 w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen, hasNeg=hasNeg, percent=0.05)

    pairs_train, labels_train, classifer_labels_train, \
    pairs_test, labels_test, classifer_labels_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    char_vob, char_id2char, char_W, c2v_k,\
    target_vob, target_id2word, type_W, type_k,\
    posi_W, posi_k,\
    max_s, max_posi, max_c = pickle.load(open(datafile, 'rb'))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x2_tag = np.asarray(pairs_train[1], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs_train[2], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs_train[3], dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")
    # train_y_classifer = np.asarray(classifer_labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x2_tag]
    inputs_train_y = [train_y]

    nn_model = SelectModel(modelname,
                           wordvocabsize=len(word_vob),
                           tagvocabsize=len(target_vob),
                           posivocabsize=max_posi+1,
                           charvocabsize=len(char_vob)+1,
                           word_W=word_W, posi_W=posi_W, tag_W=type_W, char_W=char_W,
                           input_sent_lenth=max_s,
                           w2v_k=w2v_k, posi2v_k=max_posi+1, tag2v_k=type_k, c2v_k=c2v_k, tag_k=type_k,
                           batch_size=batch_size)

    for inum in range(1, 3):

        modelfile = "./model/" + modelname + "__" + datafname + "__" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                            resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                                resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

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

