"""shim — atomic.data.tradfi (NOT PORTED — stubs)."""


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return lambda *a, **k: []
