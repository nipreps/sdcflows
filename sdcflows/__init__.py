"""SDCflows - :abbr:`SDC (susceptibility distortion correction)` by DUMMIES, for dummies."""
__packagename__ = "sdcflows"
__copyright__ = "2022, The NiPreps developers"
try:
    from ._version import __version__
except ModuleNotFoundError:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version(__packagename__)
    except PackageNotFoundError:
        __version__ = "0+unknown"
    del version
    del PackageNotFoundError
