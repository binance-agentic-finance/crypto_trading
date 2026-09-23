"""Load a raw snapshot into aligned cells × symbols frames.

Ported from the reference study's ``run_pipeline.py`` (``load_inputs``,
``lifecycle_masks``, ``funding_panel``). On a 1d grid the behaviour is the
study's, which is what the shipped bundle and its calibration were built on;
``tests/eval/test_pipeline.py`` pins it. Two things are new:

* ``interval`` — the same pools on another fixed bar width. Pool membership is
  still decided on the daily snapshot (a month's top ten does not change inside
  a day) and broadcast to the bars of each day; lifecycle masks use the bar width.
* :func:`open_interest_panel` — open interest as-of aligned to the grid, for
  ``Panel.extras``.

Missing stays missing throughout: unknown funding is NaN, not 0; open interest
before the first observation is NaN, not the first value carried backwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import interval_ms, klines_dir, oi_dir

FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
DAY = pd.Timedelta(days=1)

__all__ = ["FIELDS", "lifecycle_masks", "load_inputs", "funding_panel", "open_interest_panel"]


def _cell(interval: str) -> pd.Timedelta:
    return pd.Timedelta(milliseconds=interval_ms(interval))


def _index_cell(index) -> pd.Timedelta:
    if len(index) > 1:
        return pd.Timedelta(index[1] - index[0])
    return DAY


def lifecycle_masks(index, columns, registry, bar: pd.Timedelta = DAY):
    """Which bars are inside each contract's own life.

    Fully elapsed bars may be inputs. Prices at a bar boundary before the exact
    delivery time remain valid exit prices; later placeholder bars do not.
    Returns ``(live_bar, live_open)``.
    """
    info = {x['symbol']: x for x in registry['symbols']}
    live_bar = pd.DataFrame(True, index=index, columns=columns)
    live_open = live_bar.copy()
    for s in columns:
        item = info[s]
        on = pd.to_datetime(item['onboardDate'], unit='ms', utc=True)
        off = pd.to_datetime(item['deliveryDate'], unit='ms', utc=True)
        live_bar[s] = (index >= on) & (index + bar <= off)
        live_open[s] = (index >= on) & (index < off)
    return live_bar, live_open


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"snapshot file missing: {path} "
                                f"(run python -m cyqnt_trd.eval.snapshot.acquire first)")
    return json.loads(path.read_text())


def load_inputs(pool: str, data_dir, *, interval: str = '1d', min_history: int = 60):
    """Prices, eligibility and a design record for ``pool`` on an ``interval`` grid.

    ``pool='current'``: the ten names of ``universes.json`` (retrospective
    selection — chosen by volume at the cut-off, then looked at backwards).
    ``pool='historical'``: the point-in-time monthly top ten by 30-day quote
    volume, ranked on the prior day, warm-up memberships from 2021-04.

    Eligible = in the pool, inside the contract lifecycle, a finite close and
    quote volume on the bar, and at least ``min_history`` observed bars.
    """
    data = Path(data_dir)
    cfg = _read_json(data / 'acquisition_config.json')
    end = pd.Timestamp(cfg['end_exclusive'])
    start = pd.Timestamp(cfg['start'])
    days = pd.date_range(start, end - DAY, freq='D')
    registry = _read_json(data / 'exchange_info.json')
    info = _read_json(data / 'universes.json')
    rebalances = 0
    if pool == 'current':
        cols = info['current_top10']
        membership = pd.DataFrame(True, index=days, columns=cols)
    elif pool == 'historical':
        allowed = {r['symbol'] for r in _read_json(data / 'daily_manifest.json')
                   if r['status'] in ('ok', 'cached')}
        q = pd.DataFrame({s: pd.read_parquet(data / 'daily' / f'{s}.parquet', columns=['ts', 'quote_volume'])
                          .set_index('ts').quote_volume for s in sorted(allowed)}).reindex(days)
        live_bar, _ = lifecycle_masks(days, q.columns, registry)
        q = q.where(live_bar)
        eligible = (q.notna().cumsum() >= 60) & q.notna()
        adv = q.rolling(30, min_periods=30).mean()
        rows = []
        rebs = pd.date_range(pd.Timestamp('2021-04-01', tz='UTC'), end, freq='MS')
        for t in rebs:
            prior = t - DAY
            snap = adv.loc[prior].where(eligible.loc[prior]).dropna().sort_values(ascending=False).head(10)
            for rank, (s, v) in enumerate(snap.items(), 1):
                rows.append({'rebalance': t, 'rank': rank, 'symbol': s, 'adv30_usdt': float(v)})
        hist = pd.DataFrame(rows)
        cols = sorted(hist.symbol.unique())
        membership = pd.DataFrame(False, index=days, columns=cols)
        for k, t in enumerate(rebs):
            stop = rebs[k + 1] if k + 1 < len(rebs) else end
            names = hist.loc[hist.rebalance == t, 'symbol'].tolist()
            membership.loc[(days >= t) & (days < stop), names] = True
        rebalances = int(hist.rebalance.nunique())
    else:
        raise ValueError(f"unknown pool {pool!r}; use 'current' or 'historical'")

    bar = _cell(interval)
    if interval == '1d':
        idx = days
    else:
        if DAY % bar != pd.Timedelta(0):
            raise ValueError(f"interval {interval!r} does not tile the UTC day")
        idx = pd.date_range(start, end - bar, freq=bar)
        membership = membership.reindex(idx.floor('D')).set_axis(idx)
    source = klines_dir(data, interval)
    frames = {}
    for s in cols:
        path = source / f'{s}.parquet'
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the acquire stage for interval {interval!r}")
        frames[s] = pd.read_parquet(path).set_index('ts')
    raw = {f: pd.DataFrame({s: d[f] for s, d in frames.items()}).reindex(idx) for f in FIELDS}
    live_bar, live_open = lifecycle_masks(idx, cols, registry, bar)
    raw = {f: x.where(live_open if f == 'open' else live_bar) for f, x in raw.items()}
    available = raw['close'].notna() & raw['quote_volume'].notna()
    eligible = membership & (available.cumsum() >= min_history) & available & live_bar
    design = {
        'pool': pool, 'interval': interval, 'columns': cols, 'current_top10': info['current_top10'],
        'start': str(start), 'end_exclusive': str(end), 'min_history_bars': int(min_history),
        'rebalances': rebalances,
        'ranking_scope': ('current ten with >=60 observed bars' if pool == 'current'
                          else 'only contemporaneous monthly top ten; warmup memberships from 2021-04'),
        'selection_caveat': ('current registry incomplete for fully vanished historical symbols; current cohort is retrospective selection'
                             if pool == 'current' else
                             'current registry includes returned settling contracts but is not a complete historical securities master'),
    }
    return raw, eligible, design


def funding_panel(index, columns, data_dir):
    """Funding cash per unit of base, accumulated to the cells of ``index``.

    A settlement at time T belongs to the cell containing T. Cells with no
    settlement inside a fully paginated query window are 0; a cell with any
    unknown event (missing mark price, invalid row, acquisition-flagged gap) is
    NaN. A symbol whose pagination is not verified is NaN throughout.
    Returns ``(frame, per-symbol detail)``.
    """
    data = Path(data_dir)
    cell = _index_cell(index)
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    detail = []
    for s in columns:
        path = data / 'funding' / f'{s}.parquet'
        stamp = path.with_suffix('.json')
        if not path.exists() or not stamp.exists():
            detail.append({'symbol': s, 'status': 'missing_file'})
            continue
        meta = json.loads(stamp.read_text())
        df = pd.read_parquet(path)
        if meta.get('status') not in ('ok', 'partial', 'missing'):
            detail.append(meta)
            continue
        if meta.get('pagination_complete') is not True:
            detail.append({**meta, 'evaluation_status': 'unverified_pagination'})
            continue
        times = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
        df = df.assign(day=times.dt.floor(cell),
                       cash=pd.to_numeric(df.fundingRate) * pd.to_numeric(df.markPrice))
        # The completed pagination queried the whole requested date interval. Cells
        # without settlement events are zero; cells with an unknown event price are NA.
        grouped = df.groupby('day').cash.sum(min_count=1)
        row_valid = df.get('funding_input_valid', pd.Series(True, index=df.index)).fillna(False).astype(bool)
        unknown = set(df.loc[df.cash.isna() | ~row_valid, 'day'])
        # Acquisition also flags interval-transition ambiguity and missing events
        # that a max(previous,current interval) rule cannot identify reliably.
        # Those are recorded per UTC day, so every cell of such a day is unknown.
        for d in pd.to_datetime(meta.get('event_gap_diagnostics', {}).get('gap_affected_utc_days', []), utc=True):
            unknown.update(pd.date_range(d, d + DAY, freq=cell, inclusive='left'))
        ordered = df.assign(event_time=times).sort_values('event_time')
        intervals = pd.to_numeric(ordered.fundingIntervalHours, errors='coerce')
        gap_limit = np.maximum(intervals, intervals.shift(1)) * 3600
        gaps = ordered.event_time.diff().dt.total_seconds() > gap_limit + 1
        for pos in np.flatnonzero(gaps.to_numpy()):
            before = ordered.event_time.iloc[pos - 1] + pd.Timedelta(seconds=float(gap_limit.iloc[pos]))
            after = ordered.event_time.iloc[pos] - pd.Timedelta(seconds=1)
            if before <= after:
                unknown.update(pd.date_range(before.floor(cell), after.floor(cell), freq=cell))
        lo = pd.Timestamp(meta.get('queried_from', str(index[0])))
        hi = pd.Timestamp(meta.get('queried_through_exclusive', str(index[-1] + cell)))
        covered = (index >= lo) & (index < hi)
        out.loc[covered, s] = grouped.reindex(index[covered], fill_value=0).values
        out.loc[out.index.isin(list(unknown)), s] = np.nan
        detail.append({**meta, 'interior_gap_count': int(gaps.sum()), 'unknown_funding_days': len(unknown)})
    return out, detail


def open_interest_panel(index, columns, data_dir, *, period: str = '1h'):
    """Open interest as-of aligned to the grid, for ``Panel.extras``.

    Row ``t`` is the bar opening at ``t``; a factor at ``t`` may use what is
    known at the bar's close ``t + cell``. A snapshot stamped ``ts`` is treated
    as available only at ``ts + period`` (the conservative reading of the
    exchange's period stamp), and a value older than ``max(period, cell)`` is not
    carried forward — a gap in the feed stays a gap.

    Returns ``({'open_interest': contracts, 'open_interest_value': USDT}, detail)``.
    """
    data = Path(data_dir)
    cell = _index_cell(index)
    lag = _cell(period)
    tolerance = max(lag, cell)
    decision = pd.DataFrame({'decision': index + cell}).sort_values('decision')
    out = {'open_interest': pd.DataFrame(np.nan, index=index, columns=columns),
           'open_interest_value': pd.DataFrame(np.nan, index=index, columns=columns)}
    detail = []
    for s in columns:
        path = oi_dir(data, period) / f'{s}.parquet'
        if not path.exists():
            detail.append({'symbol': s, 'status': 'missing_file'})
            continue
        df = pd.read_parquet(path)
        df = df.assign(available=pd.to_datetime(df.timestamp, unit='ms', utc=True) + lag)
        df = df.sort_values('available')[['available', 'sumOpenInterest', 'sumOpenInterestValue']]
        joined = pd.merge_asof(decision, df, left_on='decision', right_on='available',
                               direction='backward', tolerance=tolerance)
        joined.index = joined.decision - cell
        out['open_interest'][s] = joined.sumOpenInterest.reindex(index).to_numpy()
        out['open_interest_value'][s] = joined.sumOpenInterestValue.reindex(index).to_numpy()
        detail.append({'symbol': s, 'status': 'ok', 'observations': int(len(df)),
                       'aligned_cells': int(out['open_interest'][s].notna().sum())})
    return out, detail
