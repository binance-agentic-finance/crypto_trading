"""shim — atomic.data.account"""
from cyqnt_trd.data_cli import fetch_account_balance, fetch_positions  # noqa: F401
from cyqnt_trd.compat.types import Balance, Position


def account_balance_fetch(asset="USDT", market="futures", profile=None,
                         binary="binance-cli"):
    """atomic-style — returns Balance dataclass."""
    df = fetch_account_balance(asset=asset, market=market)
    if df is None or df.empty:
        return Balance(asset=asset, free=0.0, locked=0.0, total=0.0)
    row = df.iloc[0]
    return Balance(
        asset=str(row.get("asset", asset)),
        free=float(row.get("free", 0)),
        locked=float(row.get("locked", 0)),
        total=float(row.get("total", row.get("balance", 0))),
    )


def account_info_fetch(market="futures", profile=None, binary="binance-cli"):
    """atomic-style — returns dict with balances and positions."""
    bal_df = fetch_account_balance(market=market)
    pos_df = fetch_positions(market=market)
    return {
        "balances": bal_df.to_dict(orient="records") if bal_df is not None and not bal_df.empty else [],
        "positions": pos_df.to_dict(orient="records") if pos_df is not None and not pos_df.empty else [],
    }


def position_fetch(symbol, market="futures", profile=None, binary="binance-cli"):
    """atomic-style — returns Position dataclass for a single symbol."""
    df = fetch_positions(symbol=symbol, market=market)
    if df is None or df.empty:
        return Position(
            symbol=symbol, direction="LONG", entry_price=0.0, quantity=0.0,
            unrealized_pnl=0.0, leverage=1, margin_type="ISOLATED",
        )
    row = df.iloc[0]
    qty = float(row.get("quantity", row.get("positionAmt", 0)))
    direction = "LONG" if qty >= 0 else "SHORT"
    return Position(
        symbol=str(row.get("symbol", symbol)),
        direction=direction,
        entry_price=float(row.get("entry_price", row.get("entryPrice", 0))),
        quantity=abs(qty),
        unrealized_pnl=float(row.get("unrealized_pnl", row.get("unRealizedProfit", 0))),
        leverage=int(row.get("leverage", 1) or 1),
        margin_type=str(row.get("margin_type", row.get("marginType", "ISOLATED"))).upper(),
        notional=float(row.get("notional", 0) or 0),
        mark_price=float(row.get("mark_price", row.get("markPrice", 0)) or 0),
    )
