from __future__ import print_function, division 

import ipyparallel 
import logging
import threading
import traceback
import sys
import zmq

from .globals import default_log_formatter

def pretty_tdelta(seconds):
    '''
    Prints the *seconds* in the format h mm ss
    '''
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)
        

class FunctionWrapper(object):
    def __init__(self, inner, global_vars):
        self.inner = inner
        self.glb = global_vars
        for k, v in self.glb.items():
            self.inner.__globals__[k] = v

    def __call__(self, *args, **kwargs):
        try:
            res = self.inner(*args, **kwargs)
            return (0, res)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            return (-1, (exc_type, exc_value, tb_str))


class ParallelController(object):
    '''
    A wrapper class to deal with monitoring and logging
    parallel jobs activity.

    Note that functions should -at least for now- accept only one parameter,
    (which could be a structure anyway).
    '''
    NOT_STARTED = None
    FAILED = -1
    RUNNING = 0
    SUCCESS = 1

    def __init__(self, 
                 name='ParallelController',
                 serial_fun=None,
                 args=[],
                 global_vars={}, 
                 chunksize=1,
                 logfile=None, 
                 loglevel=logging.DEBUG,
                 poll_interval=60):
        self.name = name
        self._serial_fun = serial_fun # serial function to be mapped
        self._args = args # arguments to be mapped
        self.global_vars = global_vars # globals for the function
        self.chunksize = chunksize
        self.logfile = logfile
        self.loglevel = loglevel
        self.poll_interval = poll_interval # poll for status every 
                                           # poll_interval seconds
        self.results = []
        self._status = []
        self._ok = False # flag for successfull completion
        self.chunksize
        self._client = None # ipyparallel client
        self._view = None # ipyparallel load balanced view
        self._logger = None # logger
        self._ar = None # async results
        self._fwrapper = None # function wrapper

    def get_status(self):
        return self._status

    def set_function(self, value):
        self._serial_fun = value

    def get_function(self):
        return self._serial_fun

    def set_global(self, name, val):
        self.global_vars[name] = val

    def set_args(self, args):
        self._args = args
        self._status = [ParallelController.NOT_STARTED] * len(args)
        self.results = [None] * len(self._args)

    def get_args(self):
        return self._args

    def _setup(self):
        # setup logger
        self._logger = logging.getLogger(self.name)
        # keep only stream handlers
        self._logger.handlers = [h for h in self._logger.handlers 
                                 if isinstance(h, logging.StreamHandler)] 
        
        if self.logfile is not None:
            fh = logging.FileHandler(self.logfile)
            fh.setFormatter(default_log_formatter)
            self._logger.addHandler(fh)
        self._logger.setLevel(self.loglevel)
        

        # get client and view instances, and use cloudpickle
        self._client = ipyparallel.Client(context=zmq.context())
        self._ids = self._client.ids
        self._dview = self._client[self._ids]
        #self._dview.use_cloudpickle()
        self._view = self._client.load_balanced_view(targets=self._ids)
        
        # prepare the remote function
        self._fwrapper = FunctionWrapper(self._serial_fun, self.global_vars)
        
        
    def _cleanup(self):
        self._client.close()

    def _handle_errors(self):
        failed = [i for i, x in enumerate(self._status) 
                  if x == ParallelController.FAILED]
        n_failed = len(failed)
        self._logger.error('%d tasks have failed.', n_failed)
        print('%s: %d tasks have failed.' % (self.name, n_failed), 
              file=sys.stderr)
        for cnt, i in enumerate(failed):
            if cnt > 2:
                self._logger.error('... %d more errors ...', n_failed - 3)
                print('... %d more errors ...' % (n_failed - 3), 
                      file=sys.stderr)
            self._logger.error('JOB# %d:\n %s \n' + '-'*40, i,
                               self.results[i][1][2])
            print('JOB# %d:\n %s' % (i, self.results[i][1][2]), 
                  file=sys.stderr)
            print('-'*40, file=sys.stderr)
            
    def _run(self):
        got_errors = False
        
        self._setup()
        # maps asyncronously on workers
        self._ar = self._view.map_async(self._fwrapper, self._args, 
                                        chunksize=self.chunksize)
        
        # start a thread to monitor progress
        monitor_thread = threading.Thread(target=self._monitor)
        monitor_thread.start()

        self.results = [None] * len(self._args)
        self._status = [ParallelController.RUNNING] * len(self._args)

        # collect results
        for i, result in enumerate(self._ar):
            self.results[i] = result
            if result[0] == -1:
                got_errors = True
                self._status[i] = ParallelController.FAILED
            else:
                self._status[i] = ParallelController.SUCCESS

        # close the monitor thread and print details
        monitor_thread.join()
        self._logger.info('Done. Time elapsed: %s', 
                          pretty_tdelta(self._ar.elapsed))
        
        # handle errors if any occurred
        if got_errors:
            self._handle_errors()
            raise RuntimeError('Some jobs failed. Log file: %s' % self.logfile) 
        else:
            self._ok = True

    def _monitor(self):
        while not self._ar.ready():
            n_tasks = len(self._ar)
            if self._ar.progress > 0:
                time_per_task = float(self._ar.elapsed) / self._ar.progress
                eta = (n_tasks - self._ar.progress)*time_per_task
                etastr = pretty_tdelta(eta)
            else:
                etastr = 'N/A'
            self._logger.info('Completed %d of %d tasks. Time elapsed: %s  Remaining: %s', 
                        self._ar.progress,
                        n_tasks,
                        pretty_tdelta(self._ar.elapsed),
                        etastr)
            self._ar.wait(self.poll_interval)

    def submit(self):
        if not self._serial_fun:
            raise RuntimeError('ParallelController.serial_fun not set')
        if not self._args:
            raise RuntimeError('ParallelController.args not set')
        try:
            self._setup()
            self._run()
        except Exception as e:
            raise e
        finally:
            self._cleanup()

    def success(self):
        return self._ok

    status = property(get_status, None, None)
    serial_fun = property(get_function, set_function, None)
    args = property(get_args, set_args, None)
