"""Bot-dir scaffold facade.

Every symbol below is a re-export from a single-responsibility module.
The **import path stays constant** so strategies don't break when we
split further later.

    Daemon              → daemons.SignalDaemon (compat alias)
    SignalDaemon        → daemons.signal_daemon
    SelectionDaemon     → daemons.selection_daemon
    StateBook           → state_writer
    load_bot_config     → config_loader

    build_data_adapter  → data_adapters.factory
    build_order_router  → order_routers.factory
"""
from .daemons         import Daemon, DaemonBase, SignalDaemon, SelectionDaemon
from .state_writer    import StateBook
from .config_loader   import load_bot_config
from .data_adapters   import DataAdapter, build_data_adapter
from .order_routers   import OrderRouter, build_order_router

__all__ = [
    # daemon variants
    "Daemon", "DaemonBase", "SignalDaemon", "SelectionDaemon",
    # writers / loaders
    "StateBook", "load_bot_config",
    # data + execution
    "DataAdapter", "build_data_adapter",
    "OrderRouter", "build_order_router",
]
