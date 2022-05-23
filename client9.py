from __future__ import division
from __future__ import print_function

from socketIO_client import SocketIO, LoggingNamespace
import random
from random import randrange
import time
import tensorflow as tf

import torch
import phe as paillier
# import data_diagnosis as dataprocessing
import data_fatigue as dataprocessing
import xgboost as model
import ppxgboost.tree as ptree
import ppxgboost as ppmodel
import ppgbdt as ppfullmodel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse, accuracy_score, r2_score
import threading
from random import sample,shuffle
import pandas as pd
import time


# housing 506 regression 360
# train_dataset = dataprocessing.DataSet('./data/housing.csv', 22.53, 27.47)
# test_dataset = dataprocessing.DataSet('./data/housing.csv', 22.53, 27.47)

# fatigue 360 regression
# fatigue 260
# train_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 492, 152)
# test_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 492, 152)
# # Tensile
# train_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 908, 299)
# test_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 908, 299)
# Fracture
# train_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 1644, 288)
# test_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 1644, 288)
# Hardness
# train_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 290, 90)
# test_dataset = dataprocessing.DataSet('./data/NIMS Fatigue.csv', 290, 90)

# diagnosis 569 binary 400
train_dataset = dataprocessing.DataSet('./data/data_diagnosis.csv', 0.0)
test_dataset = dataprocessing.DataSet('./data/data_diagnosis.csv', 0.0)

# breast_cancer 683 binary 480
# train_dataset = dataprocessing.DataSet('./data/breast-cancer.csv', 3.0)
# test_dataset = dataprocessing.DataSet('./data/breast-cancer.csv', 3.0)
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

enc = dict()
def encode(tree, s):
    if tree:
        if tree.leafNode == None:
            enc[s] = {
                'leafnode': False,
                'value': tree.conditionValue,
                'feature': tree.split_feature
            }
        else:
            enc[s] = {
                'leafnode': True,
                'value': tree.leafNode.predictValue
            }
        encode(tree.leftTree, s * 2)
        encode(tree.rightTree, s * 2 + 1)


dec = ptree.Tree()
def decode(dic, dec, s):
    temp = str(s)
    if dic[temp]['leafnode'] == False:
        dec.conditionValue = dic[temp]['value']
        dec.split_feature = dic[temp]['feature']
        if dic[str(s * 2)]:
            dec.leftTree = ptree.Tree()
            decode(dic, dec.leftTree, s * 2)
        if dic[str(s * 2 + 1)]:
            dec.rightTree = ptree.Tree()
            decode(dic, dec.rightTree, s * 2 + 1)
    else:
        node = ptree.LeafNode()
        node.predictValue = dic[temp]['value']
        dec.leafNode = node


class socketclient:
    def __init__(self, serverhost, serverport, mixData, poxyData):
        self.sio = SocketIO(serverhost, serverport, LoggingNamespace)
        self.finalforest = dict()
        self.keys = {}
        self.clientsId = []
        self.trainEpoch = 200
        self.epoch = 0
        self.b = 0.005
        self.currentAdj0 = 0
        self.currentAdj1 = 0
        self.currentAdj2 = 0
        self.currentAdj3 = 0
        self.currentAdj4 = 0


    def start(self):   #init parameters and build model
        self.register_handles()
        print("Starting")
        self.sio.emit("wakeup")
        self.sio.wait()

    def register_handles(self):
        def on_connect(*args):
            msg = args[0]
            self.sio.emit("connect")
            print("Connected and recieved this message")

        def on_disconnect(*args):
            print("Disconnected")
            self.sio.emit("disconnect")

        def on_ready(*args):
            print('send tree 2......')

            # train with selected dataset distribution
            dataset_train = pd.read_csv('./data/train&test/1_train.csv')
            train_data = set()
            data = list(dataset_train['train[9]'])
            dataset_test = pd.read_csv('./data/data_diagnosis/num/7_test.csv')
            test_data = list(dataset_test['test'])
            train_data = list(train_dataset.get_instances_idset() - set(train_data))
            # regression
            # label_set = [0.0]
            # classification
            label_set = train_dataset.get_label_valueset()
            y_test = []
            for i in test_data:
                y_test.append(test_dataset.get_instance(i)['diagnosis'])

            # build xgboost model with train_data
            # ppxgb = ppmodel.GBDT(sample_rate=0.9, learn_rate=0.1, max_depth=10, loss_type='regression')
            ppxgb = ppmodel.GBDT(sample_rate=0.9, learn_rate=0.1, max_depth=10, loss_type='binary-classification')
            msg = args[0]
            num = msg['flag']
            if num == 1:
                xgbtrees = dict()
                tree = ppxgb.fit(train_dataset, data, xgbtrees, 60, 1)
                encode(tree, 1)
                print(1)
                msgsend = {
                    'adj0': enc
                }
            else:
                model = msg['aggeratedAdj0']
                xgbtrees = dict()
                i = 1
                for m in model:
                    dec = ptree.Tree()
                    decode(model[m], dec, 1)
                    xgbtrees[i] = dec
                    i += 1
                tree = ppxgb.fit(train_dataset, data, xgbtrees, 60, 1)
                encode(tree, 1)
                print(2)
                msgsend = {
                    'adj0': enc
                }
            self.sio.emit("aggerateAdj0", msgsend)
            self.sio.wait()




        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('ready', on_ready)


if __name__=="__main__":
    print('client2')
    s = socketclient("127.0.0.1", 2019, -3, 3)  # mixData:-3, -7, 10, poxyData:-3, 3
    s.start()
    print("Ready")