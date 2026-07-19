# delta-farmer | https://github.com/vladkens/delta-farmer
# Copyright (c) vladkens | MIT License | Cross-exchange spread resizer
"""Cross-exchange spread resizer for any two supported venues.

Monitors the action-aware cross-exchange spread (computed from real executable BBO
prices) and, when it drops under a threshold, resizes the existing hedged BTC/ETH
basket on each venue by a USD amount you state, once, then exits.

Account selection: the tool reads each venue's existing `configs/<venue>.toml` and
uses the first account flagged `passive_farm = true` (and `enabled = true`).
See the README "Cross-exchange spread tool" section for details.

    uv run apps/spread.py status omni pacifica
    uv run apps/spread.py run    omni pacifica --action reduce   --amount 500
    uv run apps/spread.py run    omni pacifica --action increase --amount 500 --threshold 0.1 --poll 5s
    uv run apps/spread.py plan   omni pacifica --action reduce   --amount 500
    uv run apps/spread.py close  omni pacifica
"""

import argparse
import asyncio
import os
import tomllib
from decimal import Decimal

from rich import print as rprint
from rich.box import SIMPLE
from rich.table import Table

from lib.errors import AppError
from lib.logger import logger
from lib.models import AccountConfig
from lib.utils import parse_duration
from strategy.execution import close_all
from strategy.models import TradingClient
from strategy.spread import (
    SpreadState,
    Venue,
    actionable_legs,
    fetch_state,
    gather_prices_and_lots,
    plan_resize,
    run_spread_gate,
)

from .spread_registry import KNOWN_VENUES, VENUE_REGISTRY

DEFAULT_SYMBOLS = ["BTC", "ETH"]
DEFAULT_THRESHOLD = Decimal("0.1")
DEFAULT_POLL = 5.0
DEFAULT_CONFIGS_DIR = "configs"


# MARK: Config / client building


def _load_accounts(filepath: str) -> list[AccountConfig]:
    try:
        with open(filepath, "rb") as fp:
            obj = tomllib.load(fp)
    except FileNotFoundError:
        raise AppError(f"Config file not found: {filepath}")
    except tomllib.TOMLDecodeError as e:
        raise AppError(f"Invalid TOML in {filepath}: {e}")

    raw = obj.get("accounts", [])
    try:
        return [AccountConfig.model_validate(acc) for acc in raw]
    except Exception as e:
        raise AppError(f"Failed to load accounts from {filepath}: {e}")


def pick_passive_farm_account(accs: list[AccountConfig], venue: str) -> AccountConfig:
    """Return the first enabled, passive_farm-flagged account; raise if none.

    Pure function — unit-tested. Raises ValueError (not AppError) so callers/tests
    can catch it without depending on the app error type.
    """
    candidates = [a for a in accs if a.enabled and a.passive_farm]
    if not candidates:
        raise ValueError(
            f"No `passive_farm = true` (and enabled) account found for venue "
            f"{venue!r} in its config. Add the flag to an [[accounts]] block."
        )
    return candidates[0]


def _venue_config_path(venue: str, configs_dir: str) -> str:
    _, default_path = VENUE_REGISTRY[venue]
    default_name = os.path.basename(default_path)  # e.g. "omni.toml"
    return os.path.join(configs_dir, default_name)


def build_clients(venue_a: str, venue_b: str, configs_dir: str) -> dict[Venue, TradingClient]:
    """Load one passive_farm account per venue and instantiate both clients."""
    if venue_a == venue_b:
        raise AppError(f"Pick two different venues; got {venue_a!r} twice.")

    out: dict[Venue, TradingClient] = {}
    for venue in (venue_a, venue_b):
        client_cls, _ = VENUE_REGISTRY[venue]
        path = _venue_config_path(venue, configs_dir)
        accs = _load_accounts(path)
        try:
            acc = pick_passive_farm_account(accs, venue)
        except ValueError as e:
            raise AppError(str(e)) from None
        out[venue] = client_cls.from_config(acc)
    return out


# MARK: Display


def _fmt_pct(x: Decimal | None) -> str:
    return f"{x:+.4f}%" if x is not None else "n/a"


def render_spread_table(state: SpreadState, threshold: Decimal, symbols: list[str]) -> None:
    venues = list(state.bbos[symbols[0]].keys()) if symbols and state.bbos.get(symbols[0]) else []
    if not venues:
        venues = list(state.positions.keys()) or ["venue_a", "venue_b"]

    tbl = Table(box=SIMPLE, title="Cross-exchange spread")
    tbl.add_column("Symbol")
    tbl.add_column("Long venue")
    tbl.add_column("Short venue")
    tbl.add_column("Long BBO", justify="right")
    tbl.add_column("Short BBO", justify="right")
    tbl.add_column("Spread", justify="right")
    tbl.add_column(f"Gate <{threshold:.3f}%", justify="right")

    for symbol in symbols:
        pos_map = dict.fromkeys(venues, None)
        for venue in venues:
            for pos in state.positions.get(venue, []):
                if pos.symbol == symbol:
                    pos_map[venue] = pos
        long_v = next((v for v, p in pos_map.items() if p and p.side == "bid"), None)
        short_v = next((v for v, p in pos_map.items() if p and p.side == "ask"), None)

        long_bbo = state.bbos.get(symbol, {}).get(long_v) if long_v else None
        short_bbo = state.bbos.get(symbol, {}).get(short_v) if short_v else None

        def _bbo_str(b: tuple[Decimal, Decimal] | None) -> str:
            return f"{b[0]:.2f} / {b[1]:.2f}" if b else "-"

        spread = state.spreads.get(symbol)
        ok = spread is not None and spread < threshold
        verdict = (
            "[green]pass[/green]"
            if ok
            else ("[red]wait[/red]" if spread is not None else "[yellow]?[/yellow]")
        )

        tbl.add_row(
            symbol,
            long_v or "-",
            short_v or "-",
            _bbo_str(long_bbo),
            _bbo_str(short_bbo),
            _fmt_pct(spread),
            verdict,
        )

    rprint(tbl)


def render_positions(state: SpreadState, symbols: list[str]) -> None:
    tbl = Table(box=SIMPLE, title="Open positions")
    tbl.add_column("Venue")
    tbl.add_column("Symbol")
    tbl.add_column("Side")
    tbl.add_column("Size", justify="right")
    tbl.add_column("Entry", justify="right")

    for venue in state.positions:
        for pos in state.positions.get(venue, []):
            if pos.symbol not in symbols:
                continue
            color = "green" if pos.side == "bid" else "red"
            tbl.add_row(
                venue,
                pos.symbol,
                f"[{color}]{pos.side}[/{color}]",
                f"{pos.size:.6f}",
                f"{pos.entry_price:.2f}",
            )

    if tbl.row_count == 0:
        logger.info("No open positions on either venue")
    else:
        rprint(tbl)


# MARK: Commands


async def cmd_status(args: argparse.Namespace) -> None:
    clients = build_clients(args.venue_a, args.venue_b, args.configs_dir)
    symbols = args.symbols.split(",")
    state = await fetch_state(clients, symbols, args.action)
    render_spread_table(state, args.threshold, symbols)
    render_positions(state, symbols)


async def cmd_run(args: argparse.Namespace) -> None:
    clients = build_clients(args.venue_a, args.venue_b, args.configs_dir)
    symbols = args.symbols.split(",")

    def on_state(state: SpreadState) -> None:
        render_spread_table(state, args.threshold, symbols)
        render_positions(state, symbols)

    executed = await run_spread_gate(
        clients,
        action=args.action,
        amount=args.amount,
        threshold_pct=args.threshold,
        symbols=symbols,
        poll=args.poll,
        timeout=args.timeout,
        dry_run=args.dry_run,
        yes=args.yes,
        on_state=on_state if args.verbose else None,
    )
    if not executed:
        logger.warning("No resize executed this run")
        return

    final = await fetch_state(clients, symbols, args.action)
    render_positions(final, symbols)


async def cmd_close(args: argparse.Namespace) -> None:
    clients = build_clients(args.venue_a, args.venue_b, args.configs_dir)
    await close_all(list(clients.values()))


async def cmd_plan(args: argparse.Namespace) -> None:
    """Show what *would* be resized right now, using current positions + a live plan."""
    clients = build_clients(args.venue_a, args.venue_b, args.configs_dir)
    symbols = args.symbols.split(",")

    state = await fetch_state(clients, symbols, args.action)
    render_spread_table(state, args.threshold, symbols)
    render_positions(state, symbols)

    prices, lots = await gather_prices_and_lots(clients, symbols)
    plan = plan_resize({v: state.positions[v] for v in clients}, args.amount, args.action, prices, lots)

    tbl = Table(box=SIMPLE, title=f"Planned {args.action} (~${args.amount:.2f}/venue)")
    tbl.add_column("Venue")
    tbl.add_column("Symbol")
    tbl.add_column("Exec side")
    tbl.add_column("Qty", justify="right")
    tbl.add_column("USD", justify="right")
    tbl.add_column("Reduce only")
    tbl.add_column("Note")

    for leg in plan:
        tag = "skip" if leg.qty == 0 else "exec"
        tbl.add_row(
            f"[{tag}]{leg.venue}",
            leg.symbol,
            leg.side,
            f"{leg.qty:.6f}",
            f"{leg.usd:.2f}",
            "yes" if leg.reduce_only else "no",
            leg.note,
        )

    rprint(tbl)
    logger.info(f"{len(actionable_legs(plan))} actionable leg(s)")


# MARK: CLI


def _decimal_amount(v: str) -> Decimal:
    val = Decimal(v)
    if val <= 0:
        raise argparse.ArgumentTypeError("amount must be positive")
    return val


def _decimal_threshold(v: str) -> Decimal:
    val = Decimal(v)
    if val <= 0:
        raise argparse.ArgumentTypeError("threshold must be positive")
    return val


def _seconds(v: str) -> float:
    return parse_duration(v)


def _symbols(v: str) -> str:
    parts = [p.strip().upper() for p in v.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("at least one symbol required")
    return ",".join(parts)


def build_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="spread",
        description=(
            "Cross-exchange spread resizer for any two venues. "
            f"Known venues: {', '.join(KNOWN_VENUES)}."
        ),
    )
    sub = cli.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("venue_a", choices=KNOWN_VENUES, help="First venue")
        p.add_argument("venue_b", choices=KNOWN_VENUES, help="Second venue")
        p.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR, help="Dir with <venue>.toml")
        p.add_argument("--symbols", type=_symbols, default=",".join(DEFAULT_SYMBOLS))

    p_status = sub.add_parser("status", help="Print spread + positions, no trades")
    add_common(p_status)
    p_status.add_argument("--action", choices=["reduce", "increase"], default="reduce")
    p_status.add_argument("--threshold", type=_decimal_threshold, default=DEFAULT_THRESHOLD)

    p_run = sub.add_parser("run", help="Poll gate, resize once, exit")
    add_common(p_run)
    p_run.add_argument("--action", choices=["reduce", "increase"], required=True)
    p_run.add_argument("--amount", type=_decimal_amount, required=True, help="USD per exchange")
    p_run.add_argument("--threshold", type=_decimal_threshold, default=DEFAULT_THRESHOLD)
    p_run.add_argument("--poll", type=_seconds, default=DEFAULT_POLL, help="e.g. 5s, 1m")
    p_run.add_argument("--timeout", type=_seconds, default=None, help="max wait; default unlimited")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--yes", action="store_true", help="skip confirm prompt")
    p_run.add_argument("--verbose", action="store_true", help="print table on each poll")

    p_plan = sub.add_parser("plan", help="Show what would be resized right now (no orders)")
    add_common(p_plan)
    p_plan.add_argument("--action", choices=["reduce", "increase"], required=True)
    p_plan.add_argument("--amount", type=_decimal_amount, required=True)
    p_plan.add_argument("--threshold", type=_decimal_threshold, default=DEFAULT_THRESHOLD)

    p_close = sub.add_parser("close", help="Close all positions across both venues")
    add_common(p_close)

    return cli


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "status":
            await cmd_status(args)
        case "run":
            await cmd_run(args)
        case "plan":
            await cmd_plan(args)
        case "close":
            await cmd_close(args)
        case _:
            parser.print_help()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppError as e:
        logger.error(str(e))
    except KeyboardInterrupt:
        pass
