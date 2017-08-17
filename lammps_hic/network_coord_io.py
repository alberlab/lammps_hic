import sys
import traceback
import zmq
import json
import socket
try:
    import cPickle as pickle
except:
    import pickle
import logging
import os
#from .lazyio import PopulationCrdFile
import threading

from .myio import PopulationCrdFile

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='coord_server.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


def cfg_name(fname):
    return os.path.splitext(fname)[0] + '.coord_server_cfg'


class CoordServer(object):
    def __init__(self, fname, mode='r+', shape=(0, 0, 3), dtype='float32', 
                 max_memory=int(1e9), host="*", port=13457):
        self.fp = PopulationCrdFile(fname, mode, shape, dtype, max_memory)
        self.myip = socket.gethostbyname(socket.getfqdn())
        self.host = host
        self.port = port
        self.serversocket = None
        self.th = threading.Thread(target=self.run)
        config = {
            'ip' : self.myip,
            'port' : self.port
        }
        self.cfgfile = cfg_name(fname)
        with open(self.cfgfile, 'w') as f:
            json.dump(config, f)
        
    def recv(self):
        rawmessage = self.serversocket.recv()
        return pickle.loads(rawmessage)

    def send(self, obj):
        self.serversocket.send(pickle.dumps(obj))

    def start(self):
        self.th.start()

    def run(self):
        self.context = zmq.Context()
        self.serversocket = self.context.socket(zmq.REP)
        try:
            addr = 'tcp://' + self.host + ':' + str(self.port)
            self.serversocket.bind(addr)
            logger.info('server started and listening on %s' % addr)
            
            running = True
            while running:
                req = self.recv()
                logger.debug('Got request: %s', req[:2])
                try:
                    # write structure
                    if req[0] == 'ws':
                        i = req[1]
                        crd = req[2]
                        logger.debug('Write structure request: %d', i)
                        self.fp.set_struct(i, crd)
                        self.send([0])
                    # get structure
                    elif req[0] == 'gs':
                        i = req[1]
                        logger.debug('Get structure request: %d', i)
                        self.send([0, self.fp.get_struct(i)])
                    # write bead
                    elif req[0] == 'wb':
                        i = req[1]
                        crd = req[2]
                        logger.debug('Write bead request: %d', i)
                        self.fp.set_bead(i, crd)
                        self.send([0])
                    # get bead
                    elif req[0] == 'gb':
                        i = req[1]
                        logger.debug('Get bead request: %d', i)
                        self.send([0, self.fp.get_bead(i)])
                    # close request
                    elif req[0] == 'q':
                        logger.debug('Quit signal received. Exiting...')
                        self.send([0])
                        os.remove(self.cfgfile)
                        self.fp.flush()
                        self.fp.close()
                        running = False

                    else:
                        self.send([-1, 'Incorrect request %s', req[0]])
                except:
                    etype, value, tb = sys.exc_info()
                    logger.error(traceback.format_exception(etype, value, tb))
                    infostr = ''.join(traceback.format_exception(etype, value, tb))
                    if not self.serversocket.closed:
                        self.send([-1, infostr])
        finally:
            self.serversocket.setsockopt( zmq.LINGER, 0 )
            self.serversocket.close()
            self.context.term()

    def close(self):
        tmpcontext = zmq.Context()
        socket = tmpcontext.socket(zmq.REQ)
        try:
            addr = 'tcp://' + self.myip + ':' + str(self.port)
            socket.connect(addr)
            socket.send(pickle.dumps('q'))
            msg = pickle.loads(socket.recv())
            if msg[0] != 0:
                raise RuntimeError('Error on closing instance')
            self.th.join()
        finally:
            socket.setsockopt( zmq.LINGER, 0 )
            socket.close()
            tmpcontext.term()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if self.th.is_alive():
            self.close()


class CoordClient(object):
    def __init__(self, fname, wait_for_config=5):
        cfgfile = cfg_name(fname)
        config = None
        with open(cfgfile, 'r') as f:
            config = json.load(f)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        addr = 'tcp://' + config['ip'] + ':' + str(config['port'])
        self.socket.connect(addr)

    def recv(self):
        data = pickle.loads(self.socket.recv())
        if data[0] != 0:
            logger.error('Error from server: %s', data)
            raise RuntimeError('Request Error, server returned: %s' % data)
        if len(data) >= 2: # some get request
            return data[1]
        return True

    def send(self, obj):
        self.socket.send(pickle.dumps(obj))

    def get_struct(self, i):
        self.send([
            'gs',
            i,
        ])
        return self.recv()

    def set_struct(self, i, crd):
        self.send([
            'ws',
            i,
            crd,
        ])
        return self.recv()
    
    def get_bead(self, i):
        self.send([
            'gb',
            i,
        ])
        return self.recv()
    
    def set_bead(self, i, crd):
        self.send([
            'wb',
            i,
            crd,
        ])
        return self.recv()

    def shutdown(self):
        self.send([
            'q',
        ])
        return self.recv()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.socket.setsockopt( zmq.LINGER, 0 )
        self.socket.close()
        self.context.term()

        
root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
root.addHandler(logging.StreamHandler())
