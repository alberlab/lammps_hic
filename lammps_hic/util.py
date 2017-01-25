def pretty_tdelta(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)
        

def monitor_progress(routine, async_results, timeout=60):
    while not async_results.ready():
        logging.info('%s: completed %d of %d tasks. Time elapsed: %s', 
                     routine,
                     async_results.progress,
                     len(async_results),
                     pretty_tdelta(async_results.elapsed))
        async_results.wait(timeout)
