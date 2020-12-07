"""A bidirectional hashmap."""
import re

_autokey_pat = re.compile(r"^auto_(\d+)$")


class bidict(dict):
    """
    A bidirectional hashmap.

    >>> d = bidict({"a": 1, "b": 2}, c=3)
    >>> d["a"]
    1

    >>> d[1]
    'a'

    >>> 2 in d
    True

    >>> "b" in d
    True

    >>> d["d"] = 4
    >>> del d["d"]
    >>> d["d"] = 4
    >>> del d[4]
    >>> d
    {'a': 1, 'b': 2, 'c': 3}

    >>> d["d"] = "d"  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: 'd' <> 'd' is a self-mapping

    >>> d["d"]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError: 'd'

    >>> d["b"] = None  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError: 'b' is already in mapping

    >>> d[1] = None  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError: '1' is already a value in mapping

    >>> d["d"] = 1  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: '1' is already in mapping

    >>> d["d"] = "a"  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: 'a' is already a key in mapping

    >>> d["unhashable val"] = []  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: value '[]' of unhashable type: 'list'

    >>> d[list()] = 1  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: key '[]' of unhashable type: 'list'

    >>> d.add("a new value")
    'auto_00000'

    >>> d["auto_00000"]
    'a new value'

    >>> d["auto_00001"] = "another value"
    >>> d.add("a new value")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: 'a new value' is already in mapping

    >>> d.add("third value")
    'auto_00002'

    >>> d == bidict(reversed(list(d.items())))
    True

    >>> bidict({"a": 1, "b": 1})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: Bidirectional dictionary cannot contain repeated values

    >>> del d["e"]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError: 'e'

    >>> list(d)
    ['a', 'b', 'c', 'auto_00000', 'auto_00001', 'auto_00002']

    >>> list(d.values())
    [1, 2, 3, 'a new value', 'another value', 'third value']

    >>> d.clear()
    >>> d
    {}

    """

    _inverse = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inverse = {v: k for k, v in self.items()}
        if len(self) != len(self._inverse):
            raise TypeError("Bidirectional dictionary cannot contain repeated values")

    def __setitem__(self, key, value):
        if key == value:
            raise TypeError(f"'{key}' <> '{value}' is a self-mapping")

        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"value '{value}' of {exc}")
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"key '{key}' of {exc}")

        if self.__contains__(key):
            raise KeyError(
                f"'{key}' is already {'a value' * (key in self._inverse)} in mapping"
            )
        if self.__contains__(value):
            raise ValueError(
                f"'{value}' is already {'a key' * (value not in self._inverse)} in mapping"
            )

        super().__setitem__(key, value)
        self._inverse[value] = key

    def __delitem__(self, key):
        if not self.__contains__(key):
            raise KeyError(f"'{key}")

        if super().__contains__(key):
            del self._inverse[super().__getitem__(key)]
            super().__delitem__(key)
        else:
            super().__delitem__(self._inverse[key])
            del self._inverse[key]

    def __getitem__(self, key):
        if key in self._inverse:
            return self._inverse[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return key in self._inverse

    def add(self, value):
        """Insert a new value in the bidict, generating an automatic key."""
        _used = set(
            int(i.groups()[0])
            for i in [
                _autokey_pat.match(k) for k in self.keys() if k.startswith("auto_")
            ]
            if i is not None
        )
        for i in range(len(_used) + 1):
            if i not in _used:
                newkey = f"auto_{i:05d}"

        self.__setitem__(newkey, value)
        return newkey

    def clear(self):
        """Empty of all key/value pairs."""
        self._inverse.clear()
        super().clear()


class EstimatorRegistry(bidict):
    """
    A specialized :py:class:`bidict` to track :py:class:`~sdcflows.fieldmaps.FieldmapEstimation`.

    Examples
    --------
    >>> estimators = EstimatorRegistry()
    >>> _ = estimators.add(("file3.txt", "file4.txt"))
    >>> estimators.sources
    ['file3.txt', 'file4.txt']

    >>> _ = estimators.add(("file1.txt", "file2.txt"))
    >>> estimators.sources
    ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']

    >>> _ = estimators.add(("file3.txt", "file2.txt"))
    >>> estimators.sources
    ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt']

    >>> estimators.get_key("file3.txt")
    ('auto_00000', 'auto_00002')

    >>> estimators.get_key("file5.txt")
    ()

    """

    @property
    def sources(self):
        """Return a flattened list of fieldmap sources."""
        return sorted(set([el for group in self.values() for el in group]))

    def get_key(self, value):
        """Get the key(s) containing a particular value."""
        if value not in self.sources:
            return tuple()

        return tuple(sorted(k for k, v in self.items() if value in v))
