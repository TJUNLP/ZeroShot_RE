# -*- encoding:utf-8 -*-

# import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs, math, gc, random
import os.path
import numpy as np
import ProcessData_Siamese_onlySeen
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_Classifier import Model_BiLSTM_SentPair_Atloss_ed_05_Classifier


def test_model3(nn_model, tag2sentDict_test):

    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_s_all_1 = []
    data_e1_posi_all_1 = []
    data_e2_posi_all_1 = []
    char_s_all_1 = []

    data_tag_all = []
    class_labels = []

    totel_right = 0

    for tag in tag2sentDict_test.keys():
        sents = tag2sentDict_test[tag]

        for s in range(1, len(sents)):
            totel_right += 1

            for si, ty in enumerate(target_vob.values()):

                data_s, data_e1_posi, data_e2_posi, char_s = sents[s]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)
                data_tag_all.append([ty])

                data_s, data_e1_posi, data_e2_posi, char_s = sents[0]
                data_s_all_1.append(data_s)
                data_e1_posi_all_1.append(data_e1_posi)
                data_e2_posi_all_1.append(data_e2_posi)
                char_s_all_1.append(char_s)

                targetvec = np.zeros(120)
                targetvec[tag] = 1

                class_labels.append(targetvec)

    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0,
             data_s_all_1, data_e1_posi_all_1, data_e2_posi_all_1, char_s_all_1, data_tag_all]
    print(len(pairs))

    train_x1_sent = np.asarray(pairs[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs[3], dtype="int32")
    train_x2_sent = np.asarray(pairs[4], dtype="int32")
    train_x2_e1_posi = np.asarray(pairs[5], dtype="int32")
    train_x2_e2_posi = np.asarray(pairs[6], dtype="int32")
    train_x2_sent_cahr = np.asarray(pairs[7], dtype="int32")
    train_x3_sent = train_x2_sent
    train_x3_e1_posi = train_x2_e1_posi
    train_x3_e2_posi = train_x2_e2_posi
    train_x3_sent_cahr = train_x2_sent_cahr
    train_tag = np.asarray(pairs[8], dtype="int32")

    inputs_test_y0 = np.zeros(len(pairs[8]), dtype="int32")
    inputs_test_y = np.asarray(class_labels, dtype="int32")

    inputs_test_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_x2_sent, train_x2_e1_posi, train_x2_e2_posi, train_x2_sent_cahr,
                      train_x3_sent, train_x3_e1_posi, train_x3_e2_posi, train_x3_sent_cahr, train_tag]

    loss, loss1, loss2, acc = nn_model.evaluate(inputs_test_x, [inputs_test_y0, inputs_test_y], verbose=1, batch_size=1024)

    P = acc
    R = acc
    F = acc
    print('test class ... P =, R =, F = ', P, R, F)

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

        inputs_train_x, inputs_train_y = Dynamic_get_trainSet()

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_split=0.2,
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
                               verbose=1,
                               callbacks=[reduce_lr])

        print('the test result-----------------------')
        # loss, acc = nn_model.evaluate(inputs_dev_x, inputs_dev_y, batch_size=batch_size, verbose=0)
        P, R, F = test_model3(nn_model, tagDict_test)
        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(str(inum), nowepoch, earlystop, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 15:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, resultdir, w2file=''):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    # print('the test 2 result-----------------------')
    # P, R, F = test_model2(nn_model, tagDict_test)
    # print('P = ', P, 'R = ', R, 'F = ', F)
    print('the test 3 result-----------------------')
    P, R, F = test_model3(nn_model, tagDict_test)
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

    if modelname is 'Model_BiLSTM_SentPair_Atloss_ed_01_005__02_1_Classifier':
        margin = 0.1
        at_margin = 0.05
        loss_weights = {'TripletLoss': 0.2, 'CLASS': 1.}
        nn_model = Model_BiLSTM_SentPair_Atloss_ed_05_Classifier(wordvocabsize=wordvocabsize,
                                                  posivocabsize=posivocabsize,
                                                  charvocabsize=charvocabsize,
                                                    tagvocabsize=tagvocabsize,
                                                  word_W=word_W, posi_W=posi_W, char_W=char_W, tag_W=tag_W,
                                                  input_sent_lenth=input_sent_lenth,
                                                  input_maxword_length=max_c,
                                                  w2v_k=w2v_k, posi2v_k=posi2v_k, c2v_k=c2v_k, tag2v_k=tag2v_k,
                                                  batch_size=batch_size,
                                                  margin=margin, at_margin=at_margin, loss_weights=loss_weights)

    return nn_model


def Dynamic_get_trainSet(shuffle=True):


    pairs_train, labels_train = ProcessData_Siamese_onlySeen.CreateTriplet_withSoftmax(tagDict_train, shuffle, class_num=120)
    print('CreatePairs train len = ', len(pairs_train[0]), len(labels_train))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs_train[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs_train[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs_train[3], dtype="int32")
    train_x2_sent = np.asarray(pairs_train[4], dtype="int32")
    train_x2_e1_posi = np.asarray(pairs_train[5], dtype="int32")
    train_x2_e2_posi = np.asarray(pairs_train[6], dtype="int32")
    train_x2_sent_cahr = np.asarray(pairs_train[7], dtype="int32")
    train_x3_sent = np.asarray(pairs_train[8], dtype="int32")
    train_x3_e1_posi = np.asarray(pairs_train[9], dtype="int32")
    train_x3_e2_posi = np.asarray(pairs_train[10], dtype="int32")
    train_x3_sent_cahr = np.asarray(pairs_train[11], dtype="int32")
    train_tag = np.asarray(pairs_train[12], dtype="int32")

    train_y0 = np.zeros(len(labels_train), dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")
    # train_y_classifer = np.asarray(classifer_labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_x2_sent, train_x2_e1_posi, train_x2_e2_posi, train_x2_sent_cahr,
                      train_x3_sent, train_x3_e1_posi, train_x3_e2_posi, train_x3_sent_cahr, train_tag]
    inputs_train_y = [train_y0, train_y]

    return inputs_train_x, inputs_train_y


if __name__ == "__main__":

    maxlen = 100

    modelname = 'Model_BiLSTM_SentPair_Atloss_ed_05_Classifier'
    modelname = 'Model_BiLSTM_SentPair_Atloss_ed_01_005__05_1_Classifier'
    modelname = 'Model_BiLSTM_SentPair_Atloss_ed_01_005__01_1_Classifier'
    modelname = 'Model_BiLSTM_SentPair_Atloss_ed_01_005__02_1_Classifier'


    print(modelname)

    rel_prototypes_file = './data/WikiReading/rel_class_prototypes.txt.json.txt'
    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = ""
    t2v_file = './data/WikiReading/WikiReading.rel2v.by_glove.100d.txt'

    # trainfile = './data/annotated_fb__zeroshot_RE.random.train.txt'
    # testfile = './data/annotated_fb__zeroshot_RE.random.test.txt'

    # trainfile = './data/FewRel/FewRel_data.train.txt'
    # testfile = './data/FewRel/FewRel_data.test.txt'

    trainfile = './data/WikiReading/WikiReading_data.random.train.txt'
    testfile = './data/WikiReading/WikiReading_data.random.test.txt'

    # trainfile = './data/WikiReading/WikiReading_data.2.random.train.txt'
    # testfile = './data/WikiReading/WikiReading_data.3.random.test.txt'

    resultdir = "./data/result/"

    datafname = 'WikiReading_data_Siamese.WordChar.onlySeen'

    datafile = "./model/model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 512 # 1024

    retrain = False
    Test = True
    GetVec = False

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_Siamese_onlySeen.get_data(trainfile, testfile,
                                              w2v_file, c2v_file, t2v_file, datafile,
                 w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen)

    for inum in range(0, 3):

        tagDict_train, tagDict_dev, tagDict_test, \
        word_vob, word_id2word, word_W, w2v_k, \
        char_vob, char_id2char, char_W, c2v_k, \
        target_vob, target_id2word, \
        posi_W, posi_k, type_W, type_k, \
        max_s, max_posi, max_c = pickle.load(open(datafile, 'rb'))

        nn_model = SelectModel(modelname,
                               wordvocabsize=len(word_vob),
                               tagvocabsize=len(target_vob),
                               posivocabsize=max_posi + 1,
                               charvocabsize=len(char_vob),
                               word_W=word_W, posi_W=posi_W, tag_W=type_W, char_W=char_W,
                               input_sent_lenth=max_s,
                               w2v_k=w2v_k, posi2v_k=max_posi + 1, tag2v_k=type_k, c2v_k=c2v_k,
                               batch_size=batch_size)

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
            infer_e2e_model(nn_model, modelname, modelfile, resultdir, w2file=modelfile)


        del nn_model
        gc.collect()


# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

