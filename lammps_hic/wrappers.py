import multiprocessing
import logging
import threading
from .myio import pack_hms
from .globals import log_fmt
import logging.handlers


def _logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


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
            
            n_violated = pack_hms('{}/{}'.format(self.workdir, self.to_label),
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
    '''
    :Constructor Arguments:
        *from_label (string)*
            last iteration label
        *to_label (string)*
            label for the iteration to be performed
        *n_conf (int)*
            number of total structures in the population
        *workdir (string)*
            directory containing the *hms* structure files. New files
            will be also written here.
        *\*\*kwargs*
            additional arguments to be passed to the
            **lammps_hic.lammps.bulk_minimize_single_file** function. 
            See lammps_hic.lammps documentation for details

    Wrapper around multiprocessing and the lammps_hic.lammps module.
    Calling its member **run** function spawns a separate process,
    which will be responsible to connect to the ipyparallel controller
    and perform the parallel minimization of the structures.

    A modeling step requires *n_conf* files in the *workdir* directory,
    containing the starting coordinates. 
    The starting file names are expected to be in the 
    *from_label*_*n*.hms format, where *n* goes from 0 to *n_conf* - 1.

    It also requires writable **logs**, **structures**, **info** and
    **violations** directories. 

    The **run** function is syncronous and waits for the whole
    minimization to complete. A log file logs/*to_label*.log
    is updated every ~60 seconds during minimization.

    If the minimization is successful, *failed* is set to False
    and *n_violated* is set to the number of structures with
    significant violations. 

    A successful minimization will generate the following files:

    * structures/<to_label>.hss

    * violations/<to_label>.violations

    * info/<to_label>.info

    * logs/*<to_label>*.log  

    * <n_conf> files, named <workdir>/<to_label>_<n>.hms, 
      with 0 <= n < *n_conf* 

    Note that the hms files are redundant, as all the information
    is packed in the other files after the run. **They are however still 
    needed as starting configurations in the next minimization step**.

    A failed minimization will set the *failed* member to True, and 
    store the formatted traceback string in the *error* member variable.

        
    '''
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
        '''
        On a successful run, is set to the number of structures
        with violated restraints. Details of violations can be found
        in the *violations/<to_label>.violations* file
        '''
        self.error = None
        '''
        If the run fails, error is set to the formatted backtrace
        '''
        self.failed = None
        '''
        Set to True/False at the end of the run.
        '''
        self.logfile = 'logs/' + to_label + '.log'

    def run(self):
        '''
        Starts the minimization process and wait for it to complete.
        Sets the **failed** befor returning.
        '''
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_fmt))
        logger.addHandler(fh)
        logger.info('Starting Modeling Step')
        self.proc.start()
        self.proc.join()
        self.results = self.queue.get()
        logger.info('Modeling Step completed')
        logger.removeHandler(fh)
        if self.results[0] == 0:
            self.failed = False
            logger.info('completed: %d   violated: %d', self.results[1], self.results[2])
            self.n_violated = self.results[2]
            return True
        else:
            self.failed = True
            self.error = self.results[1]
            logger.error(self.results[1])
            return False
        
            
class ActivationDistancesProcess(multiprocessing.Process):

    def __init__(self, queue, **kwargs):
        super(ActivationDistancesProcess, self).__init__()
        self.queue = queue
        self.from_label = kwargs.pop('from_label')
        self.to_label = kwargs.pop('to_label')
        self.theta = kwargs.pop('theta')
        self.matrix = kwargs.pop('matrix')
        self.scatter = kwargs.pop('scatter')

    def _process(self):
        '''
        set logger and client
        '''
        import zmq
        import traceback
        from ipyparallel import Client
        
        try:
            rcl = Client(context=zmq.Context())  # Using a fresh context!
            from lammps_hic.actdist import get_actdists
            ad = get_actdists(rcl,
                             './structures/%s.hss' % self.from_label,
                             self.matrix,
                             theta=self.theta,
                             last_ad='ActDist/%s.actDist' % self.from_label,
                             save_to='ActDist/%s.actDist' % self.to_label,
                             scatter=self.scatter)

            self.queue.put((0, len(ad)))

        except:
            self.queue.put((-1, traceback.format_exc()))
            
    def run(self):
        return self._process()


class ActivationDistancesStep(object):
    '''
    :Constructor Arguments:
        *from_label (string)*
            last iteration label
        *to_label (string)*
            label for the iteration to be performed
        *theta (float)*
            Considers only contact probabilities greater than theta
        *matrix (numpy matrix)*
            Matrix of contact probabilities. 
            TODO: 
            * read filename on the process
            * use a sparse matrix
        *scatter (int)*
            Level of granularity in parallel load subdivision. See
            additional explanation in the lammps.actdist documentation.
            A value ~10 should be good for most of the situations.

    Wrapper around multiprocessing and the lammps_hic.actdist module.
    Calling its member **run** function spawns a separate process,
    which will be responsible to connect to the ipyparallel controller
    and perform the parallel calculation of activation distances.

    The activation distances step requires a population structures
    file (.hss) and the last activation distances. It ouputs a log 
    and the new actdist file. See documentation for 
    lammps_hic.actdist for details on the actdist format.

    :Required files:

        * structures/<from_label>.hss
        
        * ActDist/<from_label>.actDist

        * A probability matrix file

    :Output files:

        * ActDist/<to_label>.actDist

        * logs/<to_label>.actdist.log

    
    The **run** function is syncronous and waits for the whole
    computation to complete. A log file logs/*to_label*.actdist.log
    is updated every ~60 seconds during minimization.

    If the computation is successful, *failed* is set to False
    and *n_ad* is set to the number of records in the activation
    distances file.
    '''
    def __init__(self, from_label, to_label, matrix, theta, scatter=10):
        self.queue = multiprocessing.Queue()
        self.proc = ActivationDistancesProcess(queue=self.queue,
                                               from_label=from_label,
                                               to_label=to_label,
                                               matrix=matrix,
                                               theta=theta,
                                               scatter=scatter)
        self.results = None
        self.n_ad = None
        self.error = None
        self.failed = False
        self.logfile = 'logs/' + to_label + '.actdist.log'

    def run(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_fmt))
        logger.addHandler(fh)
        logger.info('Starting Activation Distances step')
        self.proc.start()
        self.proc.join()
        self.results = self.queue.get()
        logger.info('Activation Distances step completed')
        logger.removeHandler(fh)
        if self.results[0] == 0:
            logger.info('Number of processed pairs: %d', self.results[1])
            self.n_ad = self.results[1]
            return True
        else:
            self.failed = True
            self.error = self.results[1]
            logger.error(self.results[1])
            return False
