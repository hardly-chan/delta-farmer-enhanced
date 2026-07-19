# delta-farmer | https://github.com/vladkens/delta-farmer
# Copyright (c) vladkens | MIT License | Cross-exchange spread resizer
"""Cross-exchange spread-aware basket resizer (Omni ↔ Pacifica).

Reads whatever BTC/ETH positions are currently open on each venue and resizes them
**in place, preserving each leg's side**. The spread gate is computed from the real
executable BBO prices the resize orders would transact at:

- increase: long venue buys at ask, short venue sells at bid → cost = long_ask − short_bid
- reduce:   long venue sells at bid, short venue buys at ask → cost = short_ask − long_bid

effective_spread_pct = cost / mean(buy_price, sell_price) × 100
gate passes when effective_spread_pct < threshold (a negative spread = capture, always passes).
"""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from lib.logger import logger
from lib.utils import interruptible_sleep, round_to_tick_size

from .models import Position, Side, TradingClient, opposite_side, usd_to_qty

USD_TICK = Decimal("0.01")

# Basket weights: 1.5 BTC : 1.0 ETH per venue, per user spec.
SYMBOL_WEIGHTS: dict[str, Decimal] = {"BTC": Decimal("1.5"), "ETH": Decimal("1.0")}

Action = str  # "reduce" | "increase"
Venue = str  # "omni" | "pacifica" (or client.exchange)


# MARK: Pure helpers


def leg_usd_amounts(
    total_usd: Decimal, weights: Mapping[str, Decimal] = SYMBOL_WEIGHTS
) -> dict[str, Decimal]:
    """Split a per-exchange USD total across symbols by weight.

    $500 with {BTC: 1.5, ETH: 1.0} → {BTC: 300.00, ETH: 200.00}.
    """
    if not weights:
        raise ValueError("at least one symbol weight is required")

    total_weight = sum(weights.values(), Decimal(0))
    if total_weight <= 0:
        raise ValueError("total weight must be positive")

    out: dict[str, Decimal] = {}
    for symbol, weight in weights.items():
        out[symbol] = round_to_tick_size(total_usd * weight / total_weight, USD_TICK)
    return out


def _bbo_for_side(bbo: tuple[Decimal, Decimal], side: Side) -> Decimal:
    """Return best bid or best ask from a (bid, ask) pair."""
    bid, ask = bbo
    return bid if side == "bid" else ask


def effective_spread(
    action: Action,
    long_venue_bbo: tuple[Decimal, Decimal],
    short_venue_bbo: tuple[Decimal, Decimal],
) -> Decimal:
    """Cross-exchange spread in % using the executable prices each leg would fill at.

    increase: long venue buys at ask, short venue sells at bid.
    reduce:   long venue sells at bid, short venue buys at ask.

    Returns cost / mean(executable prices) × 100. Negative ⇒ capture.
    """
    long_bid, long_ask = long_venue_bbo
    short_bid, short_ask = short_venue_bbo

    if action == "increase":
        buy_price, sell_price = long_ask, short_bid
    elif action == "reduce":
        buy_price, sell_price = short_ask, long_bid
    else:
        raise ValueError(f"unknown action {action!r}, expected 'reduce' or 'increase'")

    mid = (buy_price + sell_price) / 2
    if mid <= 0:
        return Decimal(0)
    return (buy_price - sell_price) / mid * 100


def _venue_for_side(positions: Mapping[Venue, Position | None], side: Side) -> Venue | None:
    """Find the venue whose position matches the given side (first match)."""
    for venue, pos in positions.items():
        if pos is not None and pos.side == side:
            return venue
    return None


def compute_spreads(
    action: Action,
    positions_by_symbol: Mapping[str, Mapping[Venue, Position | None]],
    bbos_by_symbol: Mapping[str, Mapping[Venue, tuple[Decimal, Decimal]]],
) -> dict[str, Decimal | None]:
    """Per-symbol effective spread %, keyed by symbol.

    Returns None for a symbol where the long/short venue cannot be determined from
    positions, or where a BBO is missing.
    """
    out: dict[str, Decimal | None] = {}
    for symbol, venues_pos in positions_by_symbol.items():
        long_venue = _venue_for_side(venues_pos, "bid")
        short_venue = _venue_for_side(venues_pos, "ask")
        bbos = bbos_by_symbol.get(symbol, {})

        if long_venue is None or short_venue is None:
            out[symbol] = None
            continue
        if long_venue == short_venue:  # not actually hedged across venues
            out[symbol] = None
            continue
        long_bbo = bbos.get(long_venue)
        short_bbo = bbos.get(short_venue)
        if long_bbo is None or short_bbo is None:
            out[symbol] = None
            continue

        out[symbol] = effective_spread(action, long_bbo, short_bbo)
    return out


@dataclass(frozen=True)
class PlannedLeg:
    venue: Venue
    symbol: str
    side: Side  # execution side for the new market order
    qty: Decimal
    usd: Decimal
    reduce_only: bool
    pos_side: Side | None  # existing position side, None if no position
    note: str = ""  # skip/flag reason when qty == 0


def _existing_position(
    positions_by_venue: Mapping[Venue, Sequence[Position]], venue: Venue, symbol: str
) -> Position | None:
    for pos in positions_by_venue.get(venue, []):
        if pos.symbol == symbol and pos.size > 0:
            return pos
    return None


def plan_resize(
    positions_by_venue: Mapping[Venue, Sequence[Position]],
    amount_usd: Decimal,
    action: Action,
    prices: Mapping[Venue, Mapping[str, Decimal]],
    lot_sizes: Mapping[Venue, Mapping[str, Decimal]],
    weights: Mapping[str, Decimal] = SYMBOL_WEIGHTS,
) -> list[PlannedLeg]:
    """Build the resize plan for every venue × symbol.

    reduce:   side = opposite(position side), qty = min(usd_to_qty, position size), reduce_only.
    increase: side = position side, qty = usd_to_qty, reduce_only=False.

    Missing positions:
      - reduce  → skip (qty 0, note "no open position")
      - increase → flagged (qty 0, note "no open position to increase")
    """
    legs: list[PlannedLeg] = []
    per_symbol_usd = leg_usd_amounts(amount_usd, weights)

    for venue, venue_positions in positions_by_venue.items():
        venue_prices = prices.get(venue, {})
        venue_lots = lot_sizes.get(venue, {})

        for symbol, symbol_usd in per_symbol_usd.items():
            pos = _existing_position(positions_by_venue, venue, symbol)
            price = venue_prices.get(symbol)
            lot = venue_lots.get(symbol)

            if pos is None:
                note = (
                    "no open position to increase" if action == "increase" else "no open position"
                )
                legs.append(
                    PlannedLeg(
                        venue=venue,
                        symbol=symbol,
                        side="bid",
                        qty=Decimal(0),
                        usd=symbol_usd,
                        reduce_only=action == "reduce",
                        pos_side=None,
                        note=note,
                    )
                )
                continue

            if price is None or lot is None or price <= 0 or lot <= 0:
                legs.append(
                    PlannedLeg(
                        venue=venue,
                        symbol=symbol,
                        side=pos.side,
                        qty=Decimal(0),
                        usd=symbol_usd,
                        reduce_only=action == "reduce",
                        pos_side=pos.side,
                        note="missing price or lot size",
                    )
                )
                continue

            qty = usd_to_qty(symbol_usd, price, lot)
            if action == "reduce":
                exec_side: Side = opposite_side(pos.side)
                qty = min(qty, pos.size)
                reduce_only = True
                note = ""
            elif action == "increase":
                exec_side = pos.side
                reduce_only = False
                note = ""
            else:
                raise ValueError(f"unknown action {action!r}, expected 'reduce' or 'increase'")

            legs.append(
                PlannedLeg(
                    venue=venue,
                    symbol=symbol,
                    side=exec_side,
                    qty=qty,
                    usd=symbol_usd,
                    reduce_only=reduce_only,
                    pos_side=pos.side,
                    note=note,
                )
            )

    return legs


# MARK: Async orchestration


async def _bbo_or_none(client: TradingClient, symbol: str) -> tuple[Decimal, Decimal] | None:
    try:
        return await client.get_bbo(symbol)
    except Exception as e:
        logger.debug(f"{client.name}: get_bbo({symbol}) failed: {e}")
        return None


async def _price_or_none(client: TradingClient, symbol: str) -> Decimal | None:
    try:
        return await client.get_price(symbol)
    except Exception as e:
        logger.debug(f"{client.name}: get_price({symbol}) failed: {e}")
        return None


async def _lot_or_none(client: TradingClient, symbol: str) -> Decimal | None:
    try:
        return await client.get_lot_size(symbol)
    except Exception as e:
        logger.debug(f"{client.name}: get_lot_size({symbol}) failed: {e}")
        return None


async def _positions_or_empty(client: TradingClient) -> list[Position]:
    try:
        return await client.positions()
    except Exception as e:
        logger.debug(f"{client.name}: positions() failed: {e}")
        return []


@dataclass
class SpreadState:
    bbos: dict[str, dict[Venue, tuple[Decimal, Decimal]]]  # symbol -> venue -> (bid, ask)
    positions: dict[Venue, list[Position]]
    spreads: dict[str, Decimal | None]  # symbol -> effective spread % (None = unknown)


async def _auth_client(client: TradingClient) -> None:
    """Trigger auth once per client before parallel data fetches.

    Some clients (e.g. Omni with Cloudflare) solve challenges sequentially.
    A single authenticated call caches the token so parallel calls within the
    TTL window reuse it without re-challenging.
    """
    await client.balance()


async def fetch_state(
    clients: Mapping[Venue, TradingClient],
    symbols: Sequence[str],
    action: Action,
) -> SpreadState:
    """Warmup then gather BBos and positions across venues, compute action-aware spreads."""
    venues = list(clients)

    await asyncio.gather(*[c.warmup() for c in clients.values()], return_exceptions=True)
    # Auth sequentially per venue to avoid parallel CF challenges (especially Omni).
    await asyncio.gather(*[_auth_client(c) for c in clients.values()], return_exceptions=True)

    bbos: dict[str, dict[Venue, tuple[Decimal, Decimal]]] = {s: {} for s in symbols}
    positions: dict[Venue, list[Position]] = {}

    async def gather_venue(venue: Venue) -> None:
        client = clients[venue]
        bbo_vals, pos_vals = await asyncio.gather(
            asyncio.gather(*[_bbo_or_none(client, s) for s in symbols]),
            _positions_or_empty(client),
        )
        for symbol, bbo in zip(symbols, bbo_vals):
            if bbo is not None:
                bbos[symbol][venue] = bbo
        positions[venue] = pos_vals

    await asyncio.gather(*[gather_venue(v) for v in venues])

    # Build positions_by_symbol: {symbol: {venue: Position | None}}
    positions_by_symbol: dict[str, dict[Venue, Position | None]] = {
        s: dict.fromkeys(venues, None) for s in symbols
    }
    for venue, venue_positions in positions.items():
        for pos in venue_positions:
            if pos.symbol in positions_by_symbol:
                positions_by_symbol[pos.symbol][venue] = pos

    spreads = compute_spreads(action, positions_by_symbol, bbos)
    return SpreadState(bbos=bbos, positions=positions, spreads=spreads)


async def gather_prices_and_lots(
    clients: Mapping[Venue, TradingClient], symbols: Sequence[str]
) -> tuple[dict[Venue, dict[str, Decimal]], dict[Venue, dict[str, Decimal]]]:
    """Fetch mid prices and lot sizes per venue×symbol (used for sizing)."""
    # Re-auth in case the balance TTL expired since fetch_state.
    await asyncio.gather(*[_auth_client(c) for c in clients.values()], return_exceptions=True)

    prices: dict[Venue, dict[str, Decimal]] = {}
    lots: dict[Venue, dict[str, Decimal]] = {}

    async def gather_venue(venue: Venue) -> None:
        client = clients[venue]
        price_vals, lot_vals = await asyncio.gather(
            asyncio.gather(*[_price_or_none(client, s) for s in symbols]),
            asyncio.gather(*[_lot_or_none(client, s) for s in symbols]),
        )
        prices[venue] = {s: p for s, p in zip(symbols, price_vals) if p is not None}
        lots[venue] = {s: lot for s, lot in zip(symbols, lot_vals) if lot is not None}

    await asyncio.gather(*[gather_venue(v) for v in clients])
    return prices, lots


def actionable_legs(plan: Sequence[PlannedLeg]) -> list[PlannedLeg]:
    """Filter to legs that will actually send an order (qty > 0)."""
    return [leg for leg in plan if leg.qty > 0]


async def execute_resize(
    clients: Mapping[Venue, TradingClient], plan: Sequence[PlannedLeg], dry_run: bool
) -> None:
    """Fire one market_order per actionable leg. Legs per venue run sequentially (rate-limit
    friendly); venues run in parallel. Reduce legs use reduce_only=True.
    """
    actionable = actionable_legs(plan)
    if not actionable:
        logger.warning("No actionable legs to execute")
        return

    by_venue: dict[Venue, list[PlannedLeg]] = {}
    for leg in actionable:
        by_venue.setdefault(leg.venue, []).append(leg)

    async def run_venue(venue: Venue, legs: list[PlannedLeg]) -> None:
        client = clients[venue]
        for leg in legs:
            tag = "DRY" if dry_run else ""
            log = logger.bind(account=client.name)
            log.info(
                f"{tag}{'reduce' if leg.reduce_only else 'increase'} {leg.side} "
                f"{leg.qty} {leg.symbol} (~${leg.usd:.2f})"
            )
            if dry_run:
                continue
            order = await client.market_order(
                leg.symbol, leg.side, leg.qty, reduce_only=leg.reduce_only
            )
            log.debug(
                f"Filled {order.filled}/{order.size} {leg.symbol} "
                f"{'@' + str(order.price) if order.price else ''}"
            )

    await asyncio.gather(*[run_venue(v, legs) for v, legs in by_venue.items()])


async def run_spread_gate(
    clients: Mapping[Venue, TradingClient],
    *,
    action: Action,
    amount: Decimal,
    threshold_pct: Decimal,
    symbols: Sequence[str],
    poll: float = 5.0,
    timeout: float | None = None,
    dry_run: bool = False,
    yes: bool = False,
    on_state: Callable[[SpreadState], None] | None = None,
) -> bool:
    """Poll the cross-exchange spread; when all symbols pass the gate, resize once and exit.

    Returns True if the resize was executed, False if the gate timed out or was cancelled.
    `on_state(state)` (if provided) is called with each fresh SpreadState for printing.
    """
    deadline = (asyncio.get_event_loop().time() + timeout) if timeout else None
    state: SpreadState | None = None

    while True:
        state = await fetch_state(clients, symbols, action)
        if on_state is not None:
            on_state(state)

        unknown = [s for s, v in state.spreads.items() if v is None]
        passing = [s for s, v in state.spreads.items() if v is not None and v < threshold_pct]
        failing = [s for s, v in state.spreads.items() if v is not None and v >= threshold_pct]

        if unknown:
            logger.warning(f"Spread unknown for {', '.join(unknown)} (missing positions or BBOs)")
        if failing:
            logger.info(
                f"Gate not met for {', '.join(failing)} "
                f"(threshold {threshold_pct:.3f}%); waiting {poll:.0f}s..."
            )

        if not unknown and not failing and len(passing) == len(symbols):
            break  # all symbols pass

        if deadline is not None and asyncio.get_event_loop().time() >= deadline:
            logger.warning(f"Spread gate timed out after {timeout:.0f}s; no trades placed")
            return False

        try:
            await interruptible_sleep(poll)
        except asyncio.CancelledError:
            logger.info("Interrupted; no trades placed")
            raise

    assert state is not None
    prices, lots = await gather_prices_and_lots(clients, symbols)
    plan = plan_resize(
        {v: state.positions[v] for v in clients}, amount, action, prices, lots, SYMBOL_WEIGHTS
    )

    logger.info(f"Spread gate passed for all symbols; planning {action} of ${amount:.2f}/venue")
    actionable = actionable_legs(plan)
    if not actionable:
        logger.warning("Nothing to resize (no actionable legs)")
        return False

    if not yes and not dry_run:
        print("\nPlanned legs:")
        for leg in plan:
            tag = "skip" if leg.qty == 0 else "exec"
            print(
                f"  [{tag}] {leg.venue:<8} {leg.side:<3} {leg.qty:>12} {leg.symbol}  ~${leg.usd:.2f}  {leg.note}"
            )  # noqa: E501
        if input("\nProceed with live orders? [y/N] ").strip().lower() != "y":
            logger.info("Aborted by user; no trades placed")
            return False

    await execute_resize(clients, plan, dry_run)
    logger.success(f"{'Dry-run ' if dry_run else ''}resize complete")
    return True
