"""Distinguish Spawner from Worker at runtime"""

_is_worker = False


def is_worker():
    """Returns true if process is a worker"""
    return _is_worker


def set_is_worker():
    """Marks the current process as a worker"""
    global _is_worker
    _is_worker = True
