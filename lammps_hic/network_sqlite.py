import sys
import traceback
import zmq
import json
import socket
try:
    import cPickle as pickle
except:
    import pickle
import os
#from .lazyio import PopulationCrdFile
import threading

import sqlite3
import time


def cfg_name(fname):
        return fname + '.tmp_cfg'

class SqliteServer(object):
    READY = 1
    FAIL = -1
    def __init__(self, dbfname, sqlsetup=None, mode='r+', host="*", port=13558):
        try:
            self.dbfname = dbfname
            if mode == 'w':
                if os.path.isfile(dbfname):
                    os.remove(dbfname)
            self._db = sqlite3.connect(dbfname)
            if sqlsetup is not None:
                cur = self._db.cursor()
                cur.execute(sqlsetup)
            self.myip = socket.gethostbyname(socket.getfqdn())
            self.host = host
            self.port = port
            self._serversocket = None
            self._listen_thread = threading.Thread(target=self.run)
            config = {
                'ip' : self.myip,
                'port' : self.port
            }
            self.cfgfile = cfg_name(dbfname)
            with open(self.cfgfile, 'w') as f:
                json.dump(config, f)
            self.status = SqliteServer.READY
            self._db.close()
        except:
            self.status = SqliteServer.FAIL
            raise

    def recv(self):
        rawmessage = self._serversocket.recv()
        return pickle.loads(rawmessage)

    def send(self, obj):
        self._serversocket.send(pickle.dumps(obj))

    def start(self):
        self._listen_thread.start()

    def run(self):
        self._db = sqlite3.connect(self.dbfname, isolation_level=None)
        if self.status != self.READY:
            raise RuntimeError('_DB setup failed')
        self._context = zmq.Context()
        self._serversocket = self._context.socket(zmq.REP)
        cur = self._db.cursor()
        try:
            addr = 'tcp://' + self.host + ':' + str(self.port)
            self._serversocket.bind(addr)
            running = True
            while running:
                req = self.recv()
                try:
                    reqtype, query, args = req 
                    # fetchall
                    if reqtype == 'fa':
                        cur.execute(query, args)
                        self.send([0, cur.fetchall()])
                    # fetchmany
                    elif reqtype == 'fo':
                        cur.execute(query, args)
                        self.send([0, cur.fetchone()])
                    elif reqtype == 'ex':
                        cur.execute(query, args)
                        self.send([0])
                    elif reqtype == 'em':
                        # executemany
                        self._db.executemany('query', args) 
                        self.send([0])
                    #quit
                    elif reqtype == 'q':
                        self.send([0])
                        os.remove(self.cfgfile)
                        running = False

                    else:
                        self.send([-1, 'Incorrect request %s', req[0]])
                except:
                    etype, value, tb = sys.exc_info()
                    infostr = ''.join(traceback.format_exception(etype, value, tb))
                    if not self._serversocket.closed:
                        self.send([-1, infostr])
        finally:
            self._db.commit()
            self._db.close()
            self._serversocket.setsockopt( zmq.LINGER, 0 )
            self._serversocket.close()
            self._context.term()

        print 'finishing listening thread for %s', self.dbfname
        sys.stdout.flush()

    def close(self):
        tmpcontext = zmq.Context()
        socket = tmpcontext.socket(zmq.REQ)
        try:
            addr = 'tcp://' + self.myip + ':' + str(self.port)
            socket.connect(addr)
            socket.send(pickle.dumps(['q', '', (),]))
            msg = pickle.loads(socket.recv())
            if msg[0] != 0:
                raise RuntimeError('Error on closing instance:', msg[1])
            self._listen_thread.join()
        finally:
            socket.setsockopt( zmq.LINGER, 0 )
            socket.close()
            tmpcontext.term()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if self._listen_thread.is_alive():
            self.close()

class SqliteClient(object):

    def __init__(self, fname, wait_for_config=5):
        cfgfile = cfg_name(fname)
        config = None
        trial = 0
        while trial < wait_for_config:
            try:
                with open(cfgfile, 'r') as f:
                    config = json.load(f)
                break
            except:
                trial += 1
                if trial >= wait_for_config:
                    raise
                time.sleep(1)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        addr = 'tcp://' + config['ip'] + ':' + str(config['port'])
        self._socket.connect(addr)

    def recv(self):
        data = pickle.loads(self._socket.recv())
        if data[0] != 0:
            raise RuntimeError('Request Error, server returned: %s' % data)
        if len(data) >= 2: # some get request
            return data[1]
        return True

    def send(self, obj):
        self._socket.send(pickle.dumps(obj))

    def fetchall(self, query, args=tuple()):
        self.send([
            'fa',
            query,
            args,
        ])
        return self.recv()

    def fetchone(self, query, args=tuple()):
        self.send([
            'fo',
            query,
            args,
        ])
        return self.recv()

    def execute(self, query, args=tuple()):
        self.send([
            'ex',
            query,
            args,
        ])
        print 'executing', query, args
        return self.recv()

    def executemany(self, query, args=[tuple()]):
        self.send([
            'em',
            query,
            args,
        ])
        return self.recv()
    
    def shutdown(self):
        self.send([
            'q',
            '',
            (),
        ])
        return self.recv()

    def close(self):
        self._socket.setsockopt( zmq.LINGER, 0 )
        self._socket.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._socket.setsockopt( zmq.LINGER, 0 )
        self._socket.close()
        self._context.term()
