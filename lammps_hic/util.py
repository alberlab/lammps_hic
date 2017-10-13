import logging
import os.path
from .globals import async_check_timeout, default_log_formatter


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

def require_vars(names, namespace=None):
    if namespace is None:
        namespace = globals()
    for n in names:
        if n not in namespace:
            raise RuntimeError('Missing %s from global variables' % n)

def resolve_templates(templates, args):
    rt = {}
    for key, val in templates.items():
        rt[key] = val.format(*args)
    return rt

def remove_if_exists(fname):
    if os.path.isfile(fname):
        os.remove(fname)

def reverse_readline(fh, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    segment = None
    offset = 0
    fh.seek(0, os.SEEK_END)
    file_size = remaining_size = fh.tell()
    while remaining_size > 0:
        offset = min(file_size, offset + buf_size)
        fh.seek(file_size - offset)
        buffer = fh.read(min(remaining_size, buf_size))
        remaining_size -= buf_size
        lines = buffer.split('\n')
        # the first line of the buffer is probably not a complete line so
        # we'll save it and append it to the last line of the next buffer
        # we read
        if segment is not None:
            # if the previous chunk starts right from the beginning of line
            # do not concact the segment to the last line of new chunk
            # instead, yield the segment first
            if buffer[-1] is not '\n':
                lines[-1] += segment
            else:
                yield segment
        segment = lines[0]
        for index in range(len(lines) - 1, 0, -1):
            if len(lines[index]):
                yield lines[index]
    # Don't yield None if the file was empty
    if segment is not None:
        yield segment

def setLogFile(fname, loglevel=logging.INFO):
    logger = logging.getLogger()
    fh = logging.FileHandler(fname)
    fh.setFormatter(default_log_formatter)
    logger.addHandler(fh)