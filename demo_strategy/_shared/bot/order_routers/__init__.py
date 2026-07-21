"""Order routers — one file per execution backend so UI can toggle each."""
from .protocol    import OrderRouter
from .bdp_kafka   import BdpBotKafkaRouter
from .atomic_lib  import AtomicLibRouter
from .dry_run     import DryRunRouter
from .factory     import build_order_router

__all__ = [
    "OrderRouter",
    "BdpBotKafkaRouter", "AtomicLibRouter", "DryRunRouter",
    "build_order_router",
]
