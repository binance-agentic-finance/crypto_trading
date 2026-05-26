"""shim — atomic.data.market_cache.

Atomic uses sqlite-backed caching with typed read/write helpers. cyqnt_trd
uses a generic in-memory dict cache. We expose atomic's typed names as
no-op stubs so callers don't crash; actual caching happens transparently
inside cyqnt_trd.data_cli.
"""


def cache_path(*args, **kwargs):
    return None


def get_conn():
    return None


def read_klines(*args, **kwargs):
    return None


def write_klines(*args, **kwargs):
    return None


def read_ticker(*args, **kwargs):
    return None


def write_tickers(*args, **kwargs):
    return None


def read_funding(*args, **kwargs):
    return None


def write_funding(*args, **kwargs):
    return None


def read_oi_history(*args, **kwargs):
    return None


def write_oi_history(*args, **kwargs):
    return None
