from __future__ import print_function, division 

import ipyparallel 
import logging
logging.basicConfig()
import threading
import multiprocessing
import traceback
import sys
import zmq
from functools import partial
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
    def __init__(self, inner, const_vars):
        self.inner = partial(inner, **const_vars)

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
                 const_vars={}, 
                 chunksize=1,
                 logfile=None, 
                 loglevel=logging.DEBUG,
                 poll_interval=60,
                 max_batch=None):
        self.name = name
        self._serial_fun = serial_fun # serial function to be mapped
        self._args = args # arguments to be mapped
        self.const_vars = const_vars # consts for the function
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
        #self._fwrapper = None # function wrapper
        self._max_batch = max_batch
        self._queue = multiprocessing.Queue()

    def get_status(self):
        return self._status

    def set_function(self, value):
        self._serial_fun = value

    def get_function(self):
        return self._serial_fun

    def set_const(self, name, val):
        self.const_vars[name] = val

    def set_args(self, args):
        self._args = args
        self._status = [ParallelController.NOT_STARTED] * len(args)
        self.results = [None] * len(self._args)

    def get_args(self):
        return self._args

    def _setup_logger(self, batch_no=None):
        # setup logger
        if batch_no is None:
            self._logger = logging.getLogger(self.name)
        else: 
            self._logger = logging.getLogger(self.name + '/batch%d' % batch_no )
        # keep only stream handlers
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        
        if self.logfile is not None:
            fh = logging.FileHandler(self.logfile)
            fh.setFormatter(default_log_formatter)
            self._logger.addHandler(fh)
        self._logger.setLevel(self.loglevel)
        
        # prepare the remote function
        #self._fwrapper = FunctionWrapper(self._serial_fun, self.const_vars)

    def _setup_ipp(self):
        # get client and view instances, and use cloudpickle
        self._client = ipyparallel.Client(context=zmq.Context())
        self._ids = self._client.ids
        self._dview = self._client[self._ids]
        #self._dview.use_cloudpickle()
        self._view = self._client.load_balanced_view(targets=self._ids)
        
    def _cleanup(self):
        if self._client:
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
            
    def _split_batches(self):
        if self._max_batch is None:
            return [(0, len(self.args))]
        else:
            return [(b*self._max_batch, 
                     min(len(self.args), (b+1)*self._max_batch)) 
                    for b in range(len(self.args)//self._max_batch + 1)]

    def _run_all(self):
        self._setup_logger()
        self.job_batches = self._split_batches()
        got_errors = False
        self.results = [None] * len(self._args)
        self._status = [ParallelController.RUNNING] * len(self._args)
        
        self._logger.info('Starting job, divided in %d batches', len(self.job_batches))
        
        tot_time = 0
        for batch_no, (batch_start, batch_end) in enumerate(self.job_batches):
            p = multiprocessing.Process(target=self._run_batch, args=(batch_no,))
            p.start()
            
            while True:
                i, result = self._queue.get()
                if i >= 0:
                    self.results[i] = result
                    if result[0] == -1:
                        got_errors = True
                        self._status[i] = ParallelController.FAILED
                    elif result:
                        self._status[i] = ParallelController.SUCCESS
                elif i == -2:
                    raise RuntimeError('Process raised error', result)
                elif i == -3: # batch finished signal
                    tot_time += result
                    break

            p.join()

    
        # handle errors if any occurred
        if got_errors:
            self._handle_errors()
            raise RuntimeError('Some jobs failed. Log file: %s' % self.logfile) 
        else:
            self._logger.info('Done. Time elapsed: %s', 
            pretty_tdelta(tot_time))
            self._ok = True



    def _run_batch(self, batch_no):
        self._setup_logger(batch_no)
        
        batch_start, batch_end = self.job_batches[batch_no]
        self._logger.info('Starting batch %d of %d: %d tasks', 
                          batch_no + 1, len(self.job_batches), 
                          batch_end - batch_start)
        self._setup_ipp()


        # maps asyncronously on workers
        fwrapper = FunctionWrapper(self._serial_fun, self.const_vars)
        self._ar = self._view.map_async(fwrapper, 
                                        self._args[batch_start:batch_end], 
                                        chunksize=self.chunksize)
    
        # start a thread to monitor progress
        self._monitor_flag = True
        monitor_thread = threading.Thread(target=self._monitor)

        monitor_thread.start()

        try:
            # collect results
            for i, r in enumerate(self._ar):
                self._queue.put((i + batch_start, r))
        except:
            self._monitor_flag = False
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self._queue.put((-2, (exc_type, exc_value, tb_str)))

        self._queue.put((-3, self._ar.elapsed))

        # close the monitor thread and print details
        self._logger.info('Done. Time elapsed: %s', 
            pretty_tdelta(self._ar.elapsed))
        
        monitor_thread.join()
        

    def _monitor(self):
        while not self._ar.ready() and self._monitor_flag:
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
            elapsed = 0
            while elapsed < self.poll_interval:
                if not self._monitor_flag:
                    break
                self._ar.wait(1)
                elapsed += 1

    def submit(self):
        if not self._serial_fun:
            raise RuntimeError('ParallelController.serial_fun not set')
        if not self._args:
            raise RuntimeError('ParallelController.args not set')
        try:
            self._run_all()
        finally:
            self._cleanup()

    def success(self):
        return self._ok

    status = property(get_status, None, None)
    serial_fun = property(get_function, set_function, None)
    args = property(get_args, set_args, None)
