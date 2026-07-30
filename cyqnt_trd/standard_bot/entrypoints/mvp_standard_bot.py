"""标准 bot 的运行入口 —— 列出 / 查契约 / 跑一轮。

一支 bot 写完之后要能被三种人用到：前端要能列出来并显示它需要什么，评审要能
不看源码就知道它被允许发什么，调度器要能起它。这个入口就是这三件事：

    # 有哪些 bot
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_standard_bot --list

    # 一支 bot 的完整契约：声明的输入（含可回放等级）+ 允许的输出
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_standard_bot \\
        --bot funding_carry_gated --describe

    # 跑一轮（默认不下单：信号只打印）
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_standard_bot \\
        --bot funding_crowding_neutral --run

    # 编译成回测：读了不可回放的源就在这里失败，而不是跑完看到一条平线
    python -m cyqnt_trd.standard_bot.entrypoints.mvp_standard_bot \\
        --bot news_catalyst_trade --check-backtest

**这里没有下单分支。** 一轮跑完输出的是 ``StandardSignal``；要真下单必须显式
经过 executor，那是另一个进程的事。默认安全的方向是"只产信号"。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行一个标准 bot")
    parser.add_argument("--bot", default=None, help="bot_id")
    parser.add_argument("--list", action="store_true", help="列出已注册的 bot")
    parser.add_argument("--describe", action="store_true",
                        help="打印这支 bot 的完整契约（输入 + 允许的输出）")
    parser.add_argument("--run", action="store_true", help="取数并跑一轮")
    parser.add_argument("--check-backtest", action="store_true",
                        help="只检查它的数据源能不能回放，不取数")
    parser.add_argument("--config", default=None, help="JSON，bot 构造参数")
    parser.add_argument("--decision-time", type=int, default=None,
                        help="epoch ms；不给就用当前时间")
    parser.add_argument("--equity", type=float, default=None)
    parser.add_argument("--module", action="append", default=[],
                        help="额外 import 的策略模块（可重复），import 即注册")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return parser


def _load_modules(paths: List[str]) -> None:
    from ..registry import load_bot_module

    for path in paths:
        registered = load_bot_module(path)
        print("loaded %s -> %s" % (path, ", ".join(registered) or "(无新增)"),
              file=sys.stderr)


def _print_list(fmt: str) -> None:
    from ..registry import list_bots

    bots = list_bots()
    if fmt == "json":
        print(json.dumps(bots, ensure_ascii=False, indent=2))
        return
    print("%-32s %-10s %-6s %s" % ("bot_id", "kind", "持仓", "允许的 intent"))
    print("-" * 100)
    for bot in bots:
        print("%-32s %-10s %-6s %s" % (
            bot["bot_id"], bot["kind"],
            "读" if bot["reads_positions"] else "不读",
            ", ".join(bot["allowed_intents"]),
        ))


def _print_describe(bot_id: str, config: Dict[str, Any], fmt: str) -> None:
    from ..registry import describe_bot

    described = describe_bot(bot_id, **config)
    if fmt == "json":
        print(json.dumps(described, ensure_ascii=False, indent=2))
        return

    bot = described["bot"]
    print("%s — %s" % (bot["bot_id"], bot["display_name"]))
    print("  %s" % bot["description"])
    print("  kind=%s  products=%s  reads_positions=%s"
          % (bot["kind"], ",".join(bot["products"]), bot["reads_positions"]))
    print()
    print("输入（%d 个声明）" % len(described["inputs"]))
    print("  %-24s %-10s %-18s %s" % ("输入名", "必需", "可回放", "形状"))
    for key, meta in described["inputs"].items():
        if meta is None:
            print("  %-24s %s" % (key, "(未知节点)"))
            continue
        print("  %-24s %-10s %-18s %s" % (
            key, "必需" if meta["required"] else "可选",
            meta["availability"], meta["schema"] or meta["emits"]))
    print()
    print("输出  schema=cyqnt.signal/v2")
    print("  允许的 intent: %s" % ", ".join(described["output"]["allowed_intents"]))
    print()
    if described["backtestable"]:
        print("可回测：是")
    else:
        print("可回测：否 —— 以下输入没有 PIT 历史：")
        for key in described["blocks_backtest"]:
            meta = described["inputs"][key]
            print("  %-24s %s" % (key, meta["pit_hazard"] or meta["availability"]))


def _check_backtest(bot_id: str, config: Dict[str, Any]) -> int:
    from ..registry import create_bot
    from ..runtime import data as data_runtime

    bot = create_bot(bot_id, **config)
    nodes = {request.node for request in bot.required_data()}
    problems = data_runtime.validate_nodes(nodes, for_backtest=True)
    if not problems:
        print("%s：全部 %d 个数据源可回放" % (bot_id, len(nodes)))
        return 0
    print("%s 不能回测，因为：" % bot_id, file=sys.stderr)
    for problem in problems:
        print("  - %s" % problem, file=sys.stderr)
    return 1


def _run(bot_id: str, config: Dict[str, Any], *, decision_time: Optional[int],
         equity: Optional[float], fmt: str) -> int:
    from ..core import DataQuality
    from ..registry import create_bot

    bot = create_bot(bot_id, **config)
    ctx = bot.fetch_context(decision_time=decision_time, equity=equity)
    signals = bot.decide_checked(ctx)

    if fmt == "json":
        print(json.dumps({
            "bot_id": bot_id,
            "decision_time": ctx.decision_time,
            "source_status": ctx.source_status,
            "warnings": ctx.warnings,
            "signals": [signal.to_dict() for signal in signals],
        }, ensure_ascii=False, indent=2, default=str))
        return 0

    degraded = ctx.degraded_inputs
    print("decision_time=%s  输入 %d 个，降级 %d 个%s" % (
        ctx.decision_time, len(ctx.source_status), len(degraded),
        ("（%s）" % ", ".join(degraded)) if degraded else ""))
    if ctx.inferred_availability:
        print("available_time 为推断值：%s" % ", ".join(ctx.inferred_availability))
    for warning in ctx.warnings[:8]:
        print("  ! %s" % warning)
    print()

    if not signals:
        # 沉默是最难排查的失败——把"跑了但没信号"和"没跑"分开说
        print("本轮 0 条信号（bot 跑了，条件未满足）")
        return 0
    print("%-10s %-14s %-9s %-6s %-6s %s" % (
        "symbol", "intent", "平仓方向", "score", "质量", "reason"))
    print("-" * 100)
    for signal in signals:
        print("%-10s %-14s %-9s %-6s %-6s %s" % (
            signal.symbol or "-", signal.intent.value,
            signal.closes_side or "-", signal.score,
            signal.data_quality.value, ", ".join(signal.reason_codes)))
    print()
    print("共 %d 条；auto_trade_eligible 全部为 false，本入口不下单。" % len(signals))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    _load_modules(args.module)
    config = json.loads(args.config) if args.config else {}

    if args.list:
        _print_list(args.format)
        return 0
    if not args.bot:
        build_parser().print_usage(sys.stderr)
        print("error: 需要 --bot（或用 --list）", file=sys.stderr)
        return 2
    if args.describe:
        _print_describe(args.bot, config, args.format)
        return 0
    if args.check_backtest:
        return _check_backtest(args.bot, config)
    if args.run:
        return _run(args.bot, config, decision_time=args.decision_time,
                    equity=args.equity, fmt=args.format)

    _print_describe(args.bot, config, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
