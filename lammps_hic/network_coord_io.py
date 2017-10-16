import sys
import traceback
import zmq
import json
try:
    import cPickle as pickle
except:
    import pickle
import logging
import os
#from .lazyio import PopulationCrdFile
import threading
import numpy as np
import subprocess

from .population_coords import PopulationCrdFile

DEFAULT_PORT_RANGE = (15000, 16000)

def cfg_name(fname):
        return fname + '.tmp_cfg'

class CoordServer(object):
    def __init__(self, fname, mode='r+', shape=(0, 0, 3), dtype='float32', 
                 max_memory='2GB', host="*", port=None, loglevel=logging.INFO):
        self.fname = fname
        self.fp = PopulationCrdFile(fname, mode, shape, dtype, max_memory)
        self.shape = self.fp.shape
        #self.myip = socket.gethostbyname(socket.getfqdn())
        self.myip = subprocess.check_output('ifconfig eno1 |' 
                        'grep -Eo \'inet (addr:)?([0-9]*\\.){3}[0-9]*\''
                        '| grep -Eo \'([0-9]*\\.){3}[0-9]*\' | '
                        'grep -v \'127.0.0.1\' | head -1', shell=True).strip()
        self.host = host
        if port is None:
            port = np.random.randint(DEFAULT_PORT_RANGE[0], 
                                     DEFAULT_PORT_RANGE[1])
        self.port = port
        self.serversocket = None
        self.th = threading.Thread(target=self.run)
        self.status = None
        self.error = None
        self._logger = logging.getLogger('CoordServer(%s)' % fname)
        self._logger.setLevel(loglevel)
        
    def recv(self):
        rawmessage = self.serversocket.recv()
        return pickle.loads(rawmessage)

    def send(self, obj):
        self.serversocket.send(pickle.dumps(obj))

    def start(self):
        self.th.start()

    def run(self):
        try:
            self.context = zmq.Context()
            self.serversocket = self.context.socket(zmq.REP)
            self.serversocket.setsockopt(zmq.LINGER, 0)
        except:
            self.status = 'fail'
            self.error = traceback.format_exception(*sys.exc_info())
        try:
            addr = 'tcp://' + self.host + ':' + str(self.port)
            self.serversocket.bind(addr)
            config = {
                'ip' : self.myip,
                'port' : self.port
            }
            self.cfgfile = cfg_name(self.fname)
            with open(self.cfgfile, 'w') as f:
                json.dump(config, f)
            
            self._logger.info('Coord server started and listening on %s' % addr)
            
            running = True
            while running:
                req = self.recv()
                self._logger.debug('Got request: %s', req[:2])
                try:
                    # write structure
                    if req[0] == 'ws':
                        i = req[1]
                        crd = req[2]
                        self._logger.debug('Write structure request: %d', i)
                        self.fp.set_struct(i, crd)
                        self.send([0])
                    # get structure
                    elif req[0] == 'gs':
                        i = req[1]
                        self._logger.debug('Get structure request: %d', i)
                        self.send([0, self.fp.get_struct(i)])
                    # write bead
                    elif req[0] == 'wb':
                        i = req[1]
                        crd = req[2]
                        self._logger.debug('Write bead request: %d', i)
                        self.fp.set_bead(i, crd)
                        self.send([0])
                    # get bead
                    elif req[0] == 'gb':
                        i = req[1]
                        self._logger.debug('Get bead request: %d', i)
                        self.send([0, self.fp.get_bead(i)])
                    # close request
                    elif req[0] == 'q':
                        self._logger.debug('Quit signal received. Exiting...')
                        self.send([0])
                        os.remove(self.cfgfile)
                        self.fp.flush()
                        self.fp.close()
                        running = False

                    else:
                        self.send([-1, 'Incorrect request %s', req[0]])
                except:
                    etype, value, tb = sys.exc_info()
                    self._logger.error(traceback.format_exception(etype, value, tb))
                    infostr = ''.join(traceback.format_exception(etype, value, tb))
                    if not self.serversocket.closed:
                        self.send([-1, infostr])
        except:
            self.status = 'fail'
            self.error = traceback.format_exception(*sys.exc_info())
            raise 
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

    def __init__(self, fname, timeout=5, loglevel=logging.INFO):
        cfgfile = cfg_name(fname)
        config = None
        with open(cfgfile, 'r') as f:
            config = json.load(f)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        addr = 'tcp://' + config['ip'] + ':' + str(config['port'])
        self.socket.connect(addr)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self._logger = logging.getLogger('CoordClient(%s)' % fname)
        self._logger.setLevel(loglevel)
        self.timeout = timeout

    def recv(self):
        if self.poller.poll(self.timeout*1000): # 10s timeout in milliseconds
            msg = self.socket.recv()
        else:
            raise IOError("Timeout processing request")

        data = pickle.loads(msg)
        if data[0] != 0:
            self._logger.error('Error from server:\n%s' % data)
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

