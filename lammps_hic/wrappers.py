import multiprocessing
import logging
from .myio import pack_hms

class ModelingStepProcess(multiprocessing.Process):
    def __init__(self, queue, **kwargs):
        super(ModelingStepProcess, self).__init__()
        self.queue = queue
        self.from_label = kwargs.pop('from_label')
        self.to_label = kwargs.pop('to_label')
        self.n_conf = kwargs.pop('n_conf')
        self.workdir = kwargs.pop('workdir')
        self.kwargs = kwargs

    def minimization_process(self):
        '''
        set logger and client
        '''
        import zmq
        import traceback
        from ipyparallel import Client
        
        try:
            rc = Client(context=zmq.Context())  # Using a fresh context!
            from lammps_hic.lammps import bulk_minimize_single_file
            n_completed, n_violated = bulk_minimize_single_file(rc, 
                                                                self.from_label, 
                                                                self.to_label, 
                                                                self.n_conf, 
                                                                workdir=self.workdir, 
                                                                **self.kwargs)
            
            pack_hms('{}/{}'.format(self.workdir, self.to_label),
                     self.n_conf,
                     'structures/%s.hss' % self.to_label,
                     'violations/%s.violations' % self.to_label,
                     'info/%s.info' % self.to_label)

            self.queue.put((0, n_completed, n_violated))

        except:
            self.queue.put((-1, traceback.format_exc()))
            
    def run(self):
        return self.minimization_process()


class ModelingStep(object):
    def __init__(self, from_label, to_label, n_conf, workdir="tmp", **kwargs):
        self.queue = multiprocessing.Queue()
        self.proc = ModelingStepProcess(queue=self.queue,
                                        from_label=from_label,
                                        to_label=to_label,
                                        n_conf=n_conf,
                                        workdir=workdir,
                                        **kwargs)
        self.results = None
        self.n_violated = None
        self.error = None
        self.failed = False
        self.logfile = 'logs/' + to_label + '.log'

    def run(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        FORMAT = "%(asctime)-15s:%(levelname)s:%(message)s"
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(fh)
        logger.info('Starting Modeling Step')
        self.proc.start()
        self.proc.join()
        self.results = self.queue.get()
        logger.info('Modeling Step completed')
        logger.removeHandler(fh)
        if self.results[0] == 0:
            logger.info('completed: %d   violated: %d', self.results[1], self.results[2])
            self.n_violated = self.results[2]
            return True
        else:
            self.failed = True
            self.error = self.results[1]
            logger.error(self.results[1])
            return False
        
            