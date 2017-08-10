import logging
from .globals import async_check_timeout


def pretty_tdelta(seconds):
    '''
    Prints the *seconds* in the format h mm ss
    '''
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)
        

def monitor_progress(routine, async_results, timeout=async_check_timeout):
    logger = logging.getLogger()
    logger.debug('monitor_progress(): checking %s every %d seconds.', routine, timeout)
    while not async_results.ready():
        n_tasks = len(async_results)
        if async_results.progress > 0:
            time_per_task = float(async_results.elapsed) / async_results.progress
            eta = (n_tasks - async_results.progress)*time_per_task
            etastr = pretty_tdelta(eta)
        else:
            etastr = 'N/A'
        logger.info('%s: completed %d of %d tasks. Time elapsed: %s  Remaining: %s', 
                    routine,
                    async_results.progress,
                    n_tasks,
                    pretty_tdelta(async_results.elapsed),
                    etastr)
        async_results.wait(timeout)


def set_remote_vals(direct_view, **kwargs):
    logger = logging.getLogger()
    for k, v in kwargs.items():
        logger.debug('setting remote vals for %s:%s on %s', k, str(v), str(direct_view.targets))
        direct_view[k] = v 


def chromosome_string_to_numeric_id(chrom):
    '''
    Transform a list of strings in numeric
    ids (from 1 to N). Multiple chromosome copies
    will have the same id
    '''
    chr_map = {}
    chrom_id = []
    hv = 0
    for s in chrom:
        z = s.replace('chr', '')
        try:
            n = chr_map[z]
        except KeyError:
            n = hv + 1
            chr_map[z] = n
            hv = n
        chrom_id.append(n)
    return chrom_id
