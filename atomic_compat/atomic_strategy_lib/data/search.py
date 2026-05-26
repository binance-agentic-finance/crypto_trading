"""shim — atomic.data.search (NOT PORTED — stubs)."""


def hotrank_fetch(*args, **kwargs):
    return []


def trending_fetch(*args, **kwargs):
    return []


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return lambda *a, **k: []
