"""
K 线 + 因子获取（单点 / 时间区间 / 以某点为 end 向前取 n 点）

参考：model-training/trading_report_signal_ranking_v1/get_klines_data_with_factor_dire.py
提供三种方式：
- 针对某个时间点：get_kline_with_factor_at_time
- 针对时间区间：get_kline_with_factor_range
- 以某时间为 end_time 向前取 n 个点：get_kline_with_factor_n_points
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# 时间转换（与 get_futures_data 一致）
# ---------------------------------------------------------------------------

def _convert_to_timestamp_ms(time_input: Union[datetime, str, int, None]) -> Optional[int]:
    if time_input is None:
        return None
    if isinstance(time_input, datetime):
        return int(time_input.timestamp() * 1000)
    if isinstance(time_input, str):
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(time_input, fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        raise ValueError(f"无法解析时间字符串: {time_input}")
    if isinstance(time_input, int):
        return time_input if time_input > 1e10 else time_input * 1000
    raise TypeError(f"不支持的时间类型: {type(time_input)}")


def _get_interval_duration_ms(interval: str) -> int:
    mapping = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    return mapping.get(interval, 60 * 1000)


# ---------------------------------------------------------------------------
# 因子计算（与 get_klines_data_with_factor_dire 一致）
# ---------------------------------------------------------------------------

def calculate_normalized_alpha_factor(
    data_slice: pd.DataFrame,
    factor_func: Callable,
    factor_name: str,
    min_required: int = 30,
    lookback_periods: int = 30,
    factor_cache: Optional[Dict[tuple, float]] = None,
    current_point_idx: Optional[int] = None,
    **factor_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    try:
        if len(data_slice) < min_required + 2:
            return None
        available_periods = len(data_slice) - min_required - 1
        if available_periods < 2:
            return None
        actual_lookback = min(lookback_periods, max(2, available_periods))
        factor_values: List[float] = []
        cache_key_base = factor_name if current_point_idx is not None else None

        for i in range(actual_lookback + 1):
            end_idx = len(data_slice) - i
            point_idx = current_point_idx - i if current_point_idx is not None else None
            if factor_cache is not None and point_idx is not None and cache_key_base:
                cache_key = (point_idx, cache_key_base)
                if cache_key in factor_cache:
                    factor_values.append(factor_cache[cache_key])
                    continue
            start_idx = max(0, end_idx - min_required - 1)
            if end_idx <= start_idx:
                factor_values.append(0.0)
                continue
            period_slice = data_slice.iloc[start_idx:end_idx]
            try:
                factor_value = factor_func(period_slice, **factor_kwargs)
                if factor_value is not None:
                    value = float(factor_value)
                    factor_values.append(value)
                    if factor_cache is not None and point_idx is not None and cache_key_base:
                        factor_cache[(point_idx, cache_key_base)] = value
                else:
                    factor_values.append(0.0)
            except Exception:
                factor_values.append(0.0)

        if len(factor_values) < 2:
            return None
        factor_array = np.array(factor_values)
        factor_mean = factor_array.mean()
        factor_std = factor_array.std()
        if factor_std == 0 or np.isnan(factor_std):
            normalized_factors = np.zeros_like(factor_array)
        else:
            normalized_factors = (factor_array - factor_mean) / factor_std
        current_normalized = float(normalized_factors[0])
        prev_normalized = float(normalized_factors[1]) if len(normalized_factors) > 1 else 0.0
        signal = "看多" if current_normalized > 0 else ("看空" if current_normalized < 0 else "中性")
        return {
            "value": current_normalized,
            "signal": signal,
            "raw_value": float(factor_values[0]) if factor_values else 0.0,
            "prev_normalized": prev_normalized,
        }
    except Exception as e:
        logging.warning("Error calculating normalized %s factor: %s", factor_name, e)
        return None


def _get_factor_configs() -> List[tuple]:
    from cyqnt_trd.blocks.factors.ma_factor import ma_factor
    from cyqnt_trd.blocks.factors.rsi_factor import rsi_factor
    from cyqnt_trd.blocks.alphas import (
        alpha1_factor,
        alpha3_factor,
        alpha7_factor,
        alpha9_factor,
        alpha11_factor,
        alpha15_factor,
        alpha17_factor,
        alpha21_factor,
        alpha23_factor,
        alpha25_factor,
        alpha29_factor,
        alpha33_factor,
        alpha34_factor,
    )
    return [
        ("ma_factor_5", ma_factor, 6, {"period": 5}, False),
        ("rsi_factor_14", rsi_factor, 16, {"period": 14}, False),
        ("alpha1", alpha1_factor, 30, {"lookback_days": 5, "stddev_period": 20, "power": 2.0}, True),
        ("alpha3", alpha3_factor, 30, {}, True),
        ("alpha7", alpha7_factor, 30, {}, True),
        ("alpha9", alpha9_factor, 30, {}, True),
        ("alpha11", alpha11_factor, 30, {}, True),
        ("alpha15", alpha15_factor, 30, {}, True),
        ("alpha17", alpha17_factor, 30, {}, True),
        ("alpha21", alpha21_factor, 30, {}, True),
        ("alpha23", alpha23_factor, 30, {}, True),
        ("alpha25", alpha25_factor, 30, {}, True),
        ("alpha29", alpha29_factor, 30, {}, True),
        ("alpha33", alpha33_factor, 30, {}, True),
        ("alpha34", alpha34_factor, 30, {}, True),
    ]


HISTORY_WINDOW = 100
# 接口单次请求 K 线数量上限，超过则分多次请求
MAX_KLINES_PER_REQUEST = 1500


def _compute_point_factors(
    data_df: pd.DataFrame,
    idx: int,
    factor_configs: List[tuple],
    factor_cache: Dict[tuple, float],
    lookback_periods: int = 30,
) -> Optional[Dict[str, Any]]:
    time_point_data = data_df.iloc[: idx + 1]
    kline_point = data_df.iloc[idx]
    open_time = int(kline_point["open_time"])
    close_time = int(kline_point["close_time"])
    open_time_str = kline_point.get(
        "open_time_str", pd.to_datetime(open_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    )
    close_time_str = kline_point.get(
        "close_time_str", pd.to_datetime(close_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    )
    point_factors: Dict[str, Any] = {}

    for factor_name, factor_func, min_req, kwargs, is_alpha in factor_configs:
        if len(time_point_data) < min_req:
            continue
        try:
            if is_alpha:
                result = calculate_normalized_alpha_factor(
                    time_point_data,
                    factor_func,
                    factor_name,
                    min_required=min_req,
                    lookback_periods=lookback_periods,
                    factor_cache=factor_cache,
                    current_point_idx=idx,
                    **kwargs,
                )
                if result:
                    point_factors[factor_name] = {
                        "value": result.get("value", 0.0),
                        "raw_value": result.get("raw_value", 0.0),
                        "signal": result.get("signal", "中性"),
                        "prev_normalized": result.get("prev_normalized", 0.0),
                    }
            else:
                cache_key = (idx, factor_name)
                if cache_key in factor_cache:
                    factor_value = factor_cache[cache_key]
                else:
                    factor_value = factor_func(time_point_data, **kwargs)
                    if factor_value is not None:
                        factor_cache[cache_key] = float(factor_value)
                if factor_value is not None:
                    signal = "看多" if factor_value > 0 else ("看空" if factor_value < 0 else "中性")
                    point_factors[factor_name] = {
                        "value": float(factor_value),
                        "raw_value": float(factor_value),
                        "signal": signal,
                    }
        except Exception as e:
            logging.debug("Failed %s at idx %s: %s", factor_name, idx, e)

    # 仅在有足够历史时输出，不进行任何 0 填充
    if idx < HISTORY_WINDOW:
        return None
    start_history_idx = idx - HISTORY_WINDOW
    end_history_idx = idx
    hist = data_df.iloc[start_history_idx:end_history_idx]
    open_price_list = hist["open_price"].astype(float).tolist()
    high_price_list = hist["high_price"].astype(float).tolist()
    low_price_list = hist["low_price"].astype(float).tolist()
    close_price_list = hist["close_price"].astype(float).tolist()
    volume_list = hist["volume"].astype(float).tolist()

    return {
        "time_index": idx,
        "open_time": open_time,
        "open_time_str": open_time_str,
        "close_time": close_time,
        "close_time_str": close_time_str,
        "open_price": float(kline_point["open_price"]),
        "high_price": float(kline_point["high_price"]),
        "low_price": float(kline_point["low_price"]),
        "close_price": float(kline_point["close_price"]),
        "volume": float(kline_point["volume"]),
        "factors": point_factors,
        "open_price_list": open_price_list,
        "high_price_list": high_price_list,
        "low_price_list": low_price_list,
        "close_price_list": close_price_list,
        "volume_list": volume_list,
    }


def _klines_to_df(klines_data: List[Any]) -> pd.DataFrame:
    formatted: List[Dict[str, Any]] = []
    for kline in klines_data:
        if isinstance(kline, list):
            open_time = int(kline[0]) if isinstance(kline[0], str) else kline[0]
            close_time = int(kline[6]) if isinstance(kline[6], str) else kline[6]
            formatted.append({
                "datetime": pd.to_datetime(open_time, unit="ms"),
                "open_time": open_time,
                "open_time_str": pd.to_datetime(open_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                "open_price": float(kline[1]),
                "high_price": float(kline[2]),
                "low_price": float(kline[3]),
                "close_price": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": close_time,
                "close_time_str": pd.to_datetime(close_time, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            })
        elif isinstance(kline, dict):
            row = dict(kline)
            if "datetime" not in row and "open_time" in row:
                row["datetime"] = pd.to_datetime(row["open_time"], unit="ms")
            formatted.append(row)
    df = pd.DataFrame(formatted)
    return df.sort_values("datetime").reset_index(drop=True)


def _fetch_klines_df(
    token: str,
    type: str,
    interval: str,
    start_time: Optional[Union[datetime, str, int]] = None,
    end_time: Optional[Union[datetime, str, int]] = None,
    limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if type == "futures":
        from cyqnt_trd.get_data.get_futures_data import get_and_save_futures_klines
        get_fn = get_and_save_futures_klines
    else:
        from cyqnt_trd.get_data.get_trending_data import get_and_save_klines_direct
        get_fn = get_and_save_klines_direct

    kwargs: Dict[str, Any] = {"symbol": token, "interval": interval, "save_csv": False, "save_json": False}
    if limit is not None:
        kwargs["limit"] = limit
    if start_time is not None:
        kwargs["start_time"] = start_time
    if end_time is not None:
        kwargs["end_time"] = end_time

    raw = get_fn(**kwargs)
    if not raw:
        return None
    return _klines_to_df(raw)


def _fetch_klines_by_range_paginated(
    token: str,
    type: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    max_per_request: int = MAX_KLINES_PER_REQUEST,
) -> Optional[pd.DataFrame]:
    """
    按时间范围拉取 K 线；若条数超过 max_per_request，则按时间分段多次请求后合并。
    返回合并去重并按 open_time 排序的 DataFrame。
    """
    interval_ms = _get_interval_duration_ms(interval)
    num_bars_approx = max(0, (end_ms - start_ms) // interval_ms)
    if num_bars_approx == 0:
        return _fetch_klines_df(token, type, interval, start_time=start_ms, end_time=end_ms)

    if type == "futures":
        from cyqnt_trd.get_data.get_futures_data import get_and_save_futures_klines
        get_fn = get_and_save_futures_klines
    else:
        from cyqnt_trd.get_data.get_trending_data import get_and_save_klines_direct
        get_fn = get_and_save_klines_direct

    if num_bars_approx <= max_per_request:
        # 小范围时仅用 end_time + limit 单次请求，避免底层 start+end 触发的多轮分页导致卡住
        request_limit = min(num_bars_approx + 20, max_per_request)
        raw = get_fn(
            symbol=token,
            interval=interval,
            end_time=end_ms,
            limit=request_limit,
            save_csv=False,
            save_json=False,
        )
        if not raw:
            return None
        df = _klines_to_df(raw)
        # 只保留 [start_ms, end_ms] 内的 K 线
        if "open_time" in df.columns:
            df = df[df["open_time"] >= start_ms].sort_values("open_time").reset_index(drop=True)
        return df

    all_raw: List[Any] = []
    current_start = start_ms
    request_count = 0
    max_chunk_requests = 1000  # 防止异常时无限循环
    while current_start < end_ms and request_count < max_chunk_requests:
        request_count += 1
        chunk_end_ms = min(current_start + max_per_request * interval_ms, end_ms)
        batch = get_fn(
            symbol=token,
            interval=interval,
            start_time=current_start,
            end_time=chunk_end_ms,
            save_csv=False,
            save_json=False,
        )
        if not batch:
            break
        all_raw.extend(batch)
        logging.info(
            "K线分页第 %s 次请求: %s ~ %s, 本批 %s 条, 累计 %s 条",
            request_count,
            pd.to_datetime(current_start, unit="ms").strftime("%Y-%m-%d %H:%M"),
            pd.to_datetime(chunk_end_ms, unit="ms").strftime("%Y-%m-%d %H:%M"),
            len(batch),
            len(all_raw),
        )
        if len(batch) < max_per_request:
            break
        current_start = chunk_end_ms
    if not all_raw:
        return None
    df = _klines_to_df(all_raw)
    # 分页边界可能重复，按 open_time 去重
    if "open_time" in df.columns:
        df = df.drop_duplicates(subset=["open_time"], keep="first").sort_values("open_time").reset_index(drop=True)
    return df


def _compute_kline_data_with_factors(
    data_df: pd.DataFrame,
    factor_configs: List[tuple],
    lookback_periods: int = 30,
) -> List[Dict[str, Any]]:
    min_required = max(c[2] for c in factor_configs)
    # 仅对“前面至少有 HISTORY_WINDOW 根真实 K 线”的点算因子，不填充 0
    start_idx = max(min_required + 2, HISTORY_WINDOW)
    factor_cache: Dict[tuple, float] = {}
    out: List[Dict[str, Any]] = []
    for idx in range(start_idx, len(data_df)):
        try:
            point = _compute_point_factors(
                data_df, idx, factor_configs, factor_cache, lookback_periods
            )
            if point is not None:
                out.append(point)
        except Exception as e:
            logging.debug("Skip point idx %s (no padding): %s", idx, e)
    return out


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def get_kline_with_factor_at_time(
    token: str,
    type: str = "futures",
    interval: str = "30m",
    at_time: Union[datetime, str, int] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    针对某个时间点：拉取包含该时间点的 K 线，并返回该时间点对应的单根 K 线及其因子。

    Args:
        token: 交易对，如 BTCUSDT
        type: futures | spot
        interval: K 线周期，如 1m, 30m, 1h
        at_time: 目标时间点（datetime/字符串/毫秒或秒时间戳）
        limit: 拉取 K 线数量（需足够覆盖 at_time 及之前历史，用于算因子），默认 200

    Returns:
        {"status": "success"|"error", "kline_data": [单点], "metadata": {...}}
        若未找到包含 at_time 的 K 线，kline_data 为空列表。
    """
    at_ms = _convert_to_timestamp_ms(at_time) if at_time is not None else None
    if at_ms is None:
        return {"status": "error", "error": "at_time is required", "kline_data": [], "metadata": {}}

    if type == "futures":
        from cyqnt_trd.get_data.get_futures_data import get_and_save_futures_klines
        get_fn = get_and_save_futures_klines
    else:
        from cyqnt_trd.get_data.get_trending_data import get_and_save_klines_direct
        get_fn = get_and_save_klines_direct

    effective_limit = min(limit, MAX_KLINES_PER_REQUEST)
    klines_data = get_fn(
        symbol=token,
        interval=interval,
        limit=effective_limit,
        end_time=at_ms + _get_interval_duration_ms(interval),
        save_csv=False,
        save_json=False,
    )
    if not klines_data:
        return {
            "status": "error",
            "error": f"No kline data for token {token}",
            "kline_data": [],
            "metadata": {"token": token, "type": type, "interval": interval, "at_time": at_ms},
        }
    data_df = _klines_to_df(klines_data)
    # 找到 open_time <= at_time 的最后一根 K 线（即包含 at_time 的那根）
    data_df = data_df.sort_values("datetime").reset_index(drop=True)
    mask = data_df["open_time"] <= at_ms
    if not mask.any():
        return {
            "status": "success",
            "kline_data": [],
            "metadata": {
                "token": token,
                "type": type,
                "interval": interval,
                "at_time": at_ms,
                "data_rows": len(data_df),
            },
        }
    valid_indices = data_df.index[mask]
    row_idx = int(valid_indices[-1])
    factor_configs = _get_factor_configs()
    min_required = max(c[2] for c in factor_configs)
    min_idx = max(min_required + 2, HISTORY_WINDOW)
    if row_idx < min_idx:
        return {
            "status": "success",
            "kline_data": [],
            "metadata": {
                "token": token,
                "type": type,
                "interval": interval,
                "at_time": at_ms,
                "reason": "Insufficient history (need at least %s bars before point)" % HISTORY_WINDOW,
            },
        }
    factor_cache: Dict[tuple, float] = {}
    point_data = _compute_point_factors(
        data_df, row_idx, factor_configs, factor_cache, lookback_periods=30
    )
    if point_data is None:
        return {
            "status": "success",
            "kline_data": [],
            "metadata": {"token": token, "type": type, "interval": interval, "at_time": at_ms},
        }
    return {
        "status": "success",
        "kline_data": [point_data],
        "metadata": {
            "token": token,
            "type": type,
            "interval": interval,
            "at_time": at_ms,
            "at_time_str": pd.to_datetime(at_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "data_rows": len(data_df),
        },
    }


def get_kline_with_factor_range(
    token: str,
    type: str = "futures",
    interval: str = "30m",
    start_time: Union[datetime, str, int] = None,
    end_time: Union[datetime, str, int] = None,
) -> Dict[str, Any]:
    """
    针对时间区间：拉取 [start_time, end_time] 内的 K 线，并返回每根 K 线的因子。

    Args:
        token: 交易对
        type: futures | spot
        interval: K 线周期
        start_time: 开始时间（必填）
        end_time: 结束时间（必填）

    Returns:
        {"status": "success"|"error", "kline_data": [...], "metadata": {...}}
    """
    start_ms = _convert_to_timestamp_ms(start_time) if start_time is not None else None
    end_ms = _convert_to_timestamp_ms(end_time) if end_time is not None else None
    if start_ms is None or end_ms is None:
        return {
            "status": "error",
            "error": "start_time and end_time are required",
            "kline_data": [],
            "metadata": {},
        }
    if start_ms >= end_ms:
        return {
            "status": "error",
            "error": "start_time must be less than end_time",
            "kline_data": [],
            "metadata": {},
        }

    data_df = _fetch_klines_by_range_paginated(
        token=token,
        type=type,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if data_df is None or data_df.empty:
        return {
            "status": "error",
            "error": f"No kline data for token {token} in range",
            "kline_data": [],
            "metadata": {"token": token, "type": type, "interval": interval},
        }
    factor_configs = _get_factor_configs()
    kline_data_with_factors = _compute_kline_data_with_factors(data_df, factor_configs)
    return {
        "status": "success",
        "kline_data": kline_data_with_factors,
        "metadata": {
            "token": token,
            "type": type,
            "interval": interval,
            "start_time": start_ms,
            "start_time_str": pd.to_datetime(start_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_ms,
            "end_time_str": pd.to_datetime(end_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "total_kline_points": len(kline_data_with_factors),
            "total_data_points": len(data_df),
        },
    }


def get_kline_with_factor_n_points(
    token: str,
    type: str = "futures",
    interval: str = "30m",
    n: int = 500,
    end_time: Optional[Union[datetime, str, int]] = None,
) -> Dict[str, Any]:
    """
    以某时间为 end_time 向前取 n 个点：不传 end_time 时用当前时间，拉取 n 根 K 线并计算因子。

    Args:
        token: 交易对
        type: futures | spot
        interval: K 线周期
        n: 向前取的 K 线根数
        end_time: 结束时间，默认 None 表示当前时间

    Returns:
        {"status": "success"|"error", "kline_data": [...], "metadata": {...}}
    """
    if end_time is None:
        end_ms = int(datetime.now().timestamp() * 1000)
    else:
        end_ms = _convert_to_timestamp_ms(end_time)
        if end_ms is None:
            return {
                "status": "error",
                "error": "Invalid end_time",
                "kline_data": [],
                "metadata": {},
            }
    interval_ms = _get_interval_duration_ms(interval)
    # 多拉 HISTORY_WINDOW 根，使最后 n 个点各有 100 根真实历史，无需填充
    fetch_count = n + HISTORY_WINDOW
    start_ms = end_ms - fetch_count * interval_ms
    # 与 get_kline_with_factor_at_time 一致：请求包含 end_ms 所在 K 线的数据，故 end_time 传 end_ms + interval_ms
    end_time_for_fetch = end_ms + interval_ms

    data_df = _fetch_klines_by_range_paginated(
        token=token,
        type=type,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_time_for_fetch,
    )
    if data_df is None or data_df.empty:
        return {
            "status": "error",
            "error": f"No kline data for token {token}",
            "kline_data": [],
            "metadata": {"token": token, "type": type, "interval": interval, "n": n},
        }
    factor_configs = _get_factor_configs()
    kline_data_with_factors = _compute_kline_data_with_factors(data_df, factor_configs)
    # 只保留 open_time <= end_ms 的点（与 at_time 语义一致：end_time 对应“包含该时刻的那根 K 线”）
    kline_data_with_factors = [p for p in kline_data_with_factors if int(p["open_time"]) <= end_ms]
    # 只返回最后 n 个点（每个点均有 100 根真实历史）
    kline_data_with_factors = kline_data_with_factors[-n:] if len(kline_data_with_factors) > n else kline_data_with_factors
    return {
        "status": "success",
        "kline_data": kline_data_with_factors,
        "metadata": {
            "token": token,
            "type": type,
            "interval": interval,
            "n": n,
            "end_time": end_ms,
            "end_time_str": pd.to_datetime(end_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": start_ms,
            "start_time_str": pd.to_datetime(start_ms, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
            "total_kline_points": len(kline_data_with_factors),
            "total_data_points": len(data_df),
        },
    }


# ---------------------------------------------------------------------------
# 测试脚本
# ---------------------------------------------------------------------------


def _value_equals(va: Any, vb: Any, rtol: float, atol: float) -> bool:
    """递归比较两值是否在数值容差内一致（支持 dict/list/数值/其它）。"""
    if isinstance(va, dict) and isinstance(vb, dict):
        if set(va.keys()) != set(vb.keys()):
            return False
        return all(_value_equals(va[k], vb[k], rtol, atol) for k in va)
    if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
        if len(va) != len(vb):
            return False
        return all(_value_equals(x, y, rtol, atol) for x, y in zip(va, vb))
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        try:
            return bool(np.isclose(float(va), float(vb), rtol=rtol, atol=atol, equal_nan=True))
        except (TypeError, ValueError):
            return va == vb
    return va == vb


def _point_dict_equals(a: Dict[str, Any], b: Dict[str, Any], rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """比较两条 K 线+因子记录是否一致（数值字段用容差比较，含嵌套 factors）。排除 time_index（两处 data_df 不同）。"""
    keys_a, keys_b = set(a.keys()), set(b.keys())
    if keys_a != keys_b:
        return False
    skip_keys = {"time_index"}  # 两处 data_df 行号不同，不参与比较
    if int(a.get("open_time", 0)) != int(b.get("open_time", 0)):
        return False
    for k in a:
        if k in skip_keys:
            continue
        if not _value_equals(a[k], b[k], rtol, atol):
            return False
    return True


def _test_at_time_equals_n_points_last():
    """
    准则：at_time 与 end_time 相同时，get_kline_with_factor_n_points 的最后一条
    应与 get_kline_with_factor_at_time 的结果完全相同。
    """
    print("=" * 60)
    print("测试: at_time 与 end_time 相同时，n_points 末条 == at_time 单条")
    print("=" * 60)
    token = "BTCUSDT"
    type_ = "futures"
    interval = "1h"
    at_time_str = "2026-01-30 8:00:00"  # 与 end_time 相同

    r_at = get_kline_with_factor_at_time(
        token=token,
        type=type_,
        interval=interval,
        at_time=at_time_str,
    )
    r_n = get_kline_with_factor_n_points(
        token=token,
        type=type_,
        interval=interval,
        n=50,
        end_time=at_time_str,
    )

    if r_at["status"] != "success":
        print("FAIL: get_kline_with_factor_at_time 失败:", r_at.get("error"))
        return
    if r_n["status"] != "success":
        print("FAIL: get_kline_with_factor_n_points 失败:", r_n.get("error"))
        return

    at_list = r_at["kline_data"]
    n_list = r_n["kline_data"]
    if not at_list:
        print("FAIL: get_kline_with_factor_at_time 返回空 kline_data（可能历史不足）")
        return
    if not n_list:
        print("FAIL: get_kline_with_factor_n_points 返回空 kline_data")
        return

    at_point = at_list[0]
    n_last = n_list[-1]
    if int(at_point["open_time"]) != int(n_last["open_time"]):
        print(
            "FAIL: open_time 不一致 at_time=%s n_points_last=%s"
            % (at_point["open_time"], n_last["open_time"])
        )
        return
    if not _point_dict_equals(at_point, n_last):
        print("FAIL: 两条记录内容不一致（数值容差内）")
        print("at_time 首条 open_time:", at_point.get("open_time"))
        print("n_points 末条 open_time:", n_last.get("open_time"))
        return
    print("PASS: at_time 单条 与 n_points 末条 完全相同（准则满足）")
    print()


def _test_get_kline_with_factor_at_time():
    """测试 get_kline_with_factor_at_time：指定时间点的 K 线+因子."""
    print("=" * 60)
    print("测试: get_kline_with_factor_at_time")
    print("=" * 60)
    # 使用一个过去的时间，保证有足够历史
    at_time = "2026-01-30 8:00:00"
    result = get_kline_with_factor_at_time(
        token="BTCUSDT",
        type="futures",
        interval="1h",
        at_time=at_time,
    )
    print("at_time:", at_time)
    print(f"now result is :{result}")
    print("status:", result["status"])
    if result["status"] == "success":
        kline_data = result["kline_data"]
        print("kline_data 条数:", len(kline_data))
        if kline_data:
            first = kline_data[0]
            print("首条 keys:", list(first.keys())[:10], "...")
            print("metadata:", result.get("metadata", {}))
    else:
        print("error:", result.get("error"))
    print()


def _test_get_kline_with_factor_range():
    """测试 get_kline_with_factor_range：时间范围内的 K 线+因子."""
    print("=" * 60)
    print("测试: get_kline_with_factor_range")
    print("=" * 60)
    start_time = "2026-01-30 00:00:00"
    end_time = "2026-01-30 8:00:00"
    result = get_kline_with_factor_range(
        token="BTCUSDT",
        type="futures",
        interval="1h",
        start_time=start_time,
        end_time=end_time,
    )
    print("start_time:", start_time, "end_time:", end_time)
    print("status:", result["status"])
    if result["status"] == "success":
        kline_data = result["kline_data"]
        print("kline_data 条数:", len(kline_data))
        if kline_data:
            first = kline_data[0]
            print("首条 open_time:", first.get("open_time"), "keys 数量:", len(first))
            print("metadata:", result.get("metadata", {}))
    else:
        print("error:", result.get("error"))
    print()


def _test_get_kline_with_factor_n_points():
    """测试 get_kline_with_factor_n_points：从 end_time 向前 n 个点."""
    import os
    import pandas as pd

    print("=" * 60)
    print("测试: get_kline_with_factor_n_points")
    print("=" * 60)
    n = 3000
    end_time = "2026-01-30 8:00:00"  # 或 None 表示当前时间
    result = get_kline_with_factor_n_points(
        token="BTCUSDT",
        type="futures",
        interval="1h",
        n=n,
        end_time=end_time,
    )
    print("n:", n, "end_time:", end_time)
    print("status:", result["status"])
    if result["status"] == "success":
        kline_data = result["kline_data"]
        print("kline_data 条数:", len(kline_data))
        if kline_data:
            first, last = kline_data[0], kline_data[-1]
            print("首条 open_time:", first.get("open_time"), "末条 open_time:", last.get("open_time"))
            print("metadata:", result.get("metadata", {}))

            # 保存为parquet格式
            save_dir = "/Users/user/Desktop/repo/crypto_trading/cyqnt_trd/tmp"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "get_kline_with_factor_n_points_result.parquet")
            try:
                # 参考data_20260130_15/BTC_futures_1h_xxx.parquet，每行为一个K线dict
                # kline_data通常为list[dict]，直接转df
                df = pd.DataFrame(kline_data)
                df.to_parquet(save_path, engine='pyarrow', index=False)
                print(f"结果已保存为parquet文件: {save_path}")
            except Exception as e:
                print(f"保存parquet结果时出错: {e}")

        else:
            print("kline_data 为空")
    else:
        print("error:", result.get("error"))
    print()


if __name__ == "__main__":
    _test_get_kline_with_factor_at_time()
    _test_get_kline_with_factor_range()
    _test_get_kline_with_factor_n_points()
    _test_at_time_equals_n_points_last()
    print("全部测试脚本执行完毕.")
