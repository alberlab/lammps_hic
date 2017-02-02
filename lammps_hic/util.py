import logging
from .globals import async_check_timeout


def pretty_tdelta(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)
        

def monitor_progress(routine, async_results, timeout=async_check_timeout):
    logger = logging.getLogger(__name__)
    logger.debug('monitor_progress(): checking %s every %d seconds.', routine, timeout)
    while not async_results.ready():
        logger.info('%s: completed %d of %d tasks. Time elapsed: %s', 
                    routine,
                    async_results.progress,
                    len(async_results),
                    pretty_tdelta(async_results.elapsed))
        async_results.wait(timeout)


def set_remote_vals(direct_view, **kwargs):
    logger = logging.getLogger(__name__)
    for k, v in kwargs.items():
        logger.debug('setting remote vals for %s:%s on %s', k, str(v), str(direct_view.targets))
        direct_view[k] = v 