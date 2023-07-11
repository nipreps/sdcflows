"""SDCflows - :abbr:`SDC (susceptibility distortion correction)` by DUMMIES, for dummies."""
from sdcflows.data import Loader

__packagename__ = "sdcflows"
__copyright__ = "2023, The NiPreps developers"

try:
    from ._version import __version__
except ModuleNotFoundError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(__packagename__).version
    except DistributionNotFound:
        __version__ = "unknown"
    del get_distribution
    del DistributionNotFound

__all__ = (
    "__version__",
    "__packagename__",
    "__copyright__",
    "load_resource",
)

load_resource = Loader(__package__)
