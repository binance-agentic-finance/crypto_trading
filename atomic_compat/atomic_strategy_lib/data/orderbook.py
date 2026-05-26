"""shim — atomic.data.orderbook"""
from cyqnt_trd.data_cli import fetch_orderbook_depth, orderbook_imbalance  # noqa: F401
from cyqnt_trd.compat.types import OrderBook, OrderBookLevel


def orderbook_fetch(symbol, market="futures", limit=20, profile=None,
                   binary="binance-cli"):
    """atomic-style — returns OrderBook dataclass."""
    df = fetch_orderbook_depth(symbol, market=market, limit=limit)
    if df is None or df.empty:
        return OrderBook(symbol=symbol, bids=[], asks=[])
    bids, asks = [], []
    for _, row in df.iterrows():
        if row.get("side") == "bid":
            bids.append(OrderBookLevel(price=float(row["price"]), quantity=float(row["quantity"])))
        elif row.get("side") == "ask":
            asks.append(OrderBookLevel(price=float(row["price"]), quantity=float(row["quantity"])))
    return OrderBook(symbol=symbol, bids=bids, asks=asks)
