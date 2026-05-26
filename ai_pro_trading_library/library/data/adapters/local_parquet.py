"""Local parquet historical data adapter.

Per `docs/architecture/data-layer-design.md` Step 6e. Reads pre-fetched
kline parquet files from disk and serves them as canonical `list[Bar]`
through the `MarketDataAdapter` Protocol — useful for offline backtests
where hitting fapi.binance.com on every run is slow / rate-limited.

Expected on-disk layout (path is configurable):

    <root>/<market_type>/<symbol>/<timeframe>.parquet

with columns: `open_time, open, high, low, close, volume, close_time,
quote_volume, trades, taker_buy_base, taker_buy_quote`.

This is the same schema `cyqnt_trd.standard_bot.data.historical` writes,
so the new repo can consume those files directly.

Pandas reads parquet via pyarrow (or fastparquet). Both are optional
extras; install with `pip install ai-pro-trading-library[data]` or
`pip install pyarrow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ai_pro_trading_library.library.core.protocols import Bar
from ai_pro_trading_library.library.data.interfaces import KlineRequest


@dataclass
class LocalParquetAdapter:
    """Read kline data from `<root>/<market>/<symbol>/<tf>.parquet`."""

    root: str | Path = "data/parquet"
    market_type: str = "futures"

    def _path(self, instrument_id: str, timeframe: str, market_type: str | None = None) -> Path:
        m = market_type or self.market_type
        return Path(self.root) / m / instrument_id.upper() / f"{timeframe}.parquet"

    def fetch_klines(self, req: KlineRequest) -> list[Bar]:
        path = self._path(req.instrument_id, req.timeframe, req.market_type)
        if not path.exists():
            raise FileNotFoundError(
                f"parquet file not found: {path} (root={self.root})"
            )
        df = self._read_parquet(path)
        if req.start_ms is not None:
            df = df[df["close_time"] >= int(req.start_ms)]
        if req.end_ms is not None:
            df = df[df["close_time"] <= int(req.end_ms)]
        if req.limit is not None and len(df) > req.limit:
            df = df.tail(int(req.limit))
        return [self._row_to_bar(row, req.instrument_id, req.timeframe) for _, row in df.iterrows()]

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                "pyarrow (or fastparquet) is required to read parquet. "
                "Install via 'pip install pyarrow' or "
                "'pip install ai-pro-trading-library[data]'."
            ) from e

    @staticmethod
    def _row_to_bar(row: pd.Series, instrument_id: str, timeframe: str) -> Bar:
        def _opt(name: str) -> float | int | None:
            val = row.get(name) if hasattr(row, "get") else None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            return val

        return Bar(
            instrument_id=instrument_id.upper(),
            timeframe=timeframe,
            timestamp=int(row["close_time"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            quote_volume=float(_opt("quote_volume")) if _opt("quote_volume") is not None else None,
            trades=int(_opt("trades")) if _opt("trades") is not None else None,
            taker_buy_base=float(_opt("taker_buy_base")) if _opt("taker_buy_base") is not None else None,
            taker_buy_quote=float(_opt("taker_buy_quote")) if _opt("taker_buy_quote") is not None else None,
            open_time=int(_opt("open_time")) if _opt("open_time") is not None else None,
            confirmed=True,
        )

    # ---------------------------------------------------------------
    # Helpers for test fixtures + offline backtest construction
    # ---------------------------------------------------------------

    def write_klines(
        self,
        bars: Iterable[Bar],
        *,
        instrument_id: str | None = None,
        timeframe: str | None = None,
        market_type: str | None = None,
    ) -> Path:
        """Write a `list[Bar]` to the parquet location for later reads.

        If `instrument_id` / `timeframe` are not given, takes them from
        the first bar.
        """
        bars_list = list(bars)
        if not bars_list:
            raise ValueError("write_klines requires at least one Bar")
        sample = bars_list[0]
        sym = (instrument_id or sample.instrument_id).upper()
        tf = timeframe or sample.timeframe
        path = self._path(sym, tf, market_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "open_time": b.open_time,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "close_time": b.timestamp,
                    "quote_volume": b.quote_volume,
                    "trades": b.trades,
                    "taker_buy_base": b.taker_buy_base,
                    "taker_buy_quote": b.taker_buy_quote,
                }
                for b in bars_list
            ]
        )
        df.to_parquet(path, index=False)
        return path


__all__ = ["LocalParquetAdapter"]
