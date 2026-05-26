"""shim — atomic.data.web3_rest (NOT PORTED — stubs)."""
from atomic_strategy_lib.data.web3 import _stub_list, _stub_dict  # noqa: F401


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return _stub_list
