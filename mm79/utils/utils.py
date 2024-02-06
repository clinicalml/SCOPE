import sys
from mm79 import EXPERIMENT_DIR, TAKEDA_LOG_DIR
import os


def str2bool(value, raise_exc=False):

    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}

    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def logging_folder():
    import socket
    hostname = socket.gethostname()
    # if "kusanagi" in hostname:
    return os.path.join(EXPERIMENT_DIR, "logs")
    # else:
    #    return TAKEDA_LOG_DIR
