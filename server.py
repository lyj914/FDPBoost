from flask import *
import flask_socketio
import socketIO_client
from socketIO_client import SocketIO, LoggingNamespace
from flask_socketio import *
import json
import codecs
import pickle
import numpy as np
import threading
import time
import copy

class socketserver:
    def __init__(self, host, port, n, k):
        self.n = n
        self.k = k
        self.host = host
        self.port = port
        self.weight = 0
        self.adj0 = 0
        self.adj1 = 0
        self.rcvNum = 0
        self.numkeys = 0
        self.responses = 0
        self.readyNum = 0
        self.respset = set()
        self.resplist = []
        self.ready_client_ids = set()
        self.recset = set()
        self.ready_client_ids_list = list()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()
        self.weight0 = 0
        self.weight1 = 0
        self.lastWeight0 = 0
        self.lastWeight1 = 0
        self.adj0 = 0
        self.adj1 = 0
        self.adj2 = 0
        self.adj3 = 0
        self.adj4 = 0
        self.flag = True
        self.xgbtrees = dict()
        self.startTime = 0
        self.endTime = 0

    def register_handles(self):

        @self.socketio.on("wakeup")
        def handle_wakeup():
            print("Recieved wakeup from", request.sid)
            self.numkeys += 1

            if self.numkeys == self.n:  # all clients are connected, let clients train theirs' model

                # self.thread = threading.Thread(target=self.boardcastHeartbeat)
                # self.thread.start()

                print('All clients connected, send ready to every clients')
                for clientId in self.ready_client_ids:
                    self.socketio.emit('connect', room=clientId)
                self.startTime = time.time()
                ready_client_ids_list = list(self.ready_client_ids)
                flag = {
                    'flag': 1
                }
                self.socketio.emit('ready', flag, room=ready_client_ids_list[0])
                self.numkeys = 0

        @self.socketio.on("connect")
        def handle_connect():
            print(request.sid, " Connected")
            self.ready_client_ids.add(request.sid)
            print('Connected devices:', self.ready_client_ids)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
            print(self.ready_client_ids)

        @self.socketio.on('aggerateAdj0')
        def handle_aggerateAdj0(*args):
            print('Get adj from client:', request.sid)
            if self.numkeys == 0:
                msg = args[0]
                self.xgbtrees[0] = msg['adj0']
                print(self.xgbtrees[0])
                self.numkeys += 1
                self.recset.add(request.sid)
                msgsend = {
                    'flag': self.numkeys+1,
                    'aggeratedAdj0': self.xgbtrees
                }
                self.recset.add(request.sid)
                ready_client_ids = self.ready_client_ids - self.recset
                self.ready_client_ids_list = list(ready_client_ids)
                self.socketio.emit('ready', msgsend, room=self.ready_client_ids_list[0])
            elif self.numkeys < self.n:
                msg = args[0]
                self.xgbtrees[self.numkeys] = msg['adj0']
                print(self.xgbtrees[self.numkeys])
                self.numkeys += 1
                if self.numkeys < self.n:
                    msgsend = {
                        'flag': self.numkeys + 1,
                        'aggeratedAdj0': self.xgbtrees
                    }
                    self.recset.add(request.sid)
                    ready_client_ids = self.ready_client_ids - self.recset
                    self.ready_client_ids_list = list(ready_client_ids)
                    self.socketio.emit('ready', msgsend, room=self.ready_client_ids_list[0])
            if self.numkeys == self.n:
                self.endTime = time.time()
                print((self.endTime - self.startTime)*16.0)



    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


if __name__=="__main__":
    server = socketserver("127.0.0.1", 2019, 5, 5)
    print("listening on 127.0.0.1:2019")
    server.start()




