"""Test miscellaneous utilities."""
import sys
from collections import namedtuple
import types
import pytest
from ..misc import get_free_mem


@pytest.mark.parametrize("retval", [None, 10])
def test_get_free_mem(monkeypatch, retval):
    """Test the get_free_mem utility."""

    def mock_func():
        if retval is None:
            raise ImportError
        return namedtuple("Mem", ("free",))(free=retval)

    psutil = types.ModuleType("psutil")
    psutil.virtual_memory = mock_func
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    assert get_free_mem() == retval
