"""Data adapters — one file per backend so UI can toggle/edit each one."""
from .protocol      import DataAdapter
from .bdp_bot       import BdpBotAdapter
from .atomic_lib    import AtomicLibAdapter
from .direct_rest   import DirectRESTAdapter
from .factory       import build_data_adapter

__all__ = [
    "DataAdapter",
    "BdpBotAdapter", "AtomicLibAdapter", "DirectRESTAdapter",
    "build_data_adapter",
]
