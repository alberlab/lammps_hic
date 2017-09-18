
# the executable to be called to run lammps
lammps_executable = 'lmp_serial_mod'

# the epsilon below energy is considered zero
float_epsilon = 1e-2

# how often print the status of async parallel jobs
async_check_timeout = 60

#format for logger
import logging
log_fmt = '[%(name)s]%(asctime)s (%(levelname)s) %(message)s'
default_log_formatter = logging.Formatter(log_fmt)


