"""shim — atomic.data.web3 (NOT PORTED — stubs only).

The web3/on-chain data sources require auth + endpoints we don't have
in cyqnt_trd. All functions return safe defaults so callers don't crash.
"""


def _stub_list(*args, **kwargs):
    return []


def _stub_dict(*args, **kwargs):
    return {}


# Most-used names — return empty lists / dicts so callers can no-op gracefully
web3_rank = _stub_list
web3_tokenized_list = _stub_list
web3_tokenized_dynamic = _stub_dict
smart_money_inflow_rank = _stub_list
social_hype_rank = _stub_list
meme_rank = _stub_list


def __getattr__(name):
    """Catch-all: any other web3 function returns a safe default."""
    if name.startswith("_"):
        raise AttributeError(name)
    return _stub_list
