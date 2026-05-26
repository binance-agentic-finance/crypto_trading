"""shim — atomic.data.http_client (NOT PORTED — stubs).

Atomic's HTTP client wraps requests for direct REST. cyqnt_trd uses
binance-cli subprocess instead, so we don't need this. Callers that
import http_client get safe defaults.
"""


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return lambda *a, **k: None
