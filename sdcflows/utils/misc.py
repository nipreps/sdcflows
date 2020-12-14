"""Basic miscelaneous utilities."""


def front(inlist):
    """
    Pop from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> front([1, 0])
    1

    >>> front("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def last(inlist):
    """
    Return the last element from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> last([1, 0])
    0

    >>> last("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[-1]
    return inlist


def get_free_mem():
    """Probe the free memory right now."""
    try:
        from psutil import virtual_memory

        return round(virtual_memory().free, 1)
    except Exception:
        return None
