from decimal import Decimal

import pytest
from pydantic import SecretStr

from apps.spread import pick_passive_farm_account
from apps.spread_registry import KNOWN_VENUES, VENUE_REGISTRY
from lib.models import AccountConfig
from strategy.models import Position, Side
from strategy.spread import (
    SYMBOL_WEIGHTS,
    USD_TICK,
    actionable_legs,
    compute_spreads,
    effective_spread,
    leg_usd_amounts,
    plan_resize,
)

D = Decimal


def _pos(symbol: str, side: Side, size: str, price: str = "100") -> Position:
    return Position(
        id=f"{symbol}_{side}", symbol=symbol, side=side, size=D(size), entry_price=D(price)
    )  # noqa: E501


# MARK: leg_usd_amounts


def test_leg_usd_amounts_default_weights_split_1_5_to_1():
    out = leg_usd_amounts(D("500"), SYMBOL_WEIGHTS)
    # BTC weight 1.5 / 2.5 = 0.6 -> 300.00; ETH weight 1.0 / 2.5 = 0.4 -> 200.00
    assert out == {"BTC": D("300.00"), "ETH": D("200.00")}


def test_leg_usd_amounts_preserves_ratio():
    out = leg_usd_amounts(D("1000"), SYMBOL_WEIGHTS)
    assert out["BTC"] / out["ETH"] == D("1.5") / D("1.0")


def test_leg_usd_amounts_rounded_to_usd_tick():
    # 1/7 would produce repeating decimals; must round cleanly to 0.01
    out = leg_usd_amounts(D("100"), {"A": D("1"), "B": D("2"), "C": D("4")})
    for v in out.values():
        assert v == (v / USD_TICK).quantize(D(1)) * USD_TICK
    assert sum(out.values()) == D("100.00")


def test_leg_usd_amounts_rejects_empty_weights():
    with pytest.raises(ValueError):
        leg_usd_amounts(D("500"), {})


def test_leg_usd_amounts_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        leg_usd_amounts(D("500"), {"BTC": D("0"), "ETH": D("0")})


# MARK: effective_spread


def test_effective_spread_increase_uses_long_ask_short_bid():
    # increase: buy at long ask 102, sell at short bid 100 → cost 2 / mid 101 = ~1.98%
    spread = effective_spread("increase", (D("100"), D("102")), (D("100"), D("103")))
    expected = (D("102") - D("100")) / ((D("102") + D("100")) / 2) * 100
    assert spread == expected
    assert spread > 0


def test_effective_spread_reduce_uses_long_bid_short_ask():
    # reduce: long sells at bid 99, short buys at ask 101 → cost 101-99=2 / mid 100 = 2%
    spread = effective_spread("reduce", (D("99"), D("101")), (D("100"), D("101")))
    expected = (D("101") - D("99")) / ((D("101") + D("99")) / 2) * 100
    assert spread == expected
    assert spread == D("2")


def test_effective_spread_zero_when_executable_prices_equal():
    # increase: long ask == short bid → 0%
    assert effective_spread("increase", (D("100"), D("100")), (D("100"), D("100"))) == D("0")
    # reduce: short ask == long bid → 0%
    assert effective_spread("reduce", (D("100"), D("100")), (D("100"), D("100"))) == D("0")


def test_effective_spread_negative_is_capture():
    # increase: long ask 100 < short bid 101 → you'd capture the gap → negative
    spread = effective_spread("increase", (D("99"), D("100")), (D("101"), D("102")))
    assert spread < 0


def test_effective_spread_rejects_unknown_action():
    with pytest.raises(ValueError):
        effective_spread("hold", (D("1"), D("1")), (D("1"), D("1")))


# MARK: compute_spreads


def test_compute_spreads_resolves_long_short_from_positions():
    positions_by_symbol = {
        "BTC": {"omni": _pos("BTC", "bid", "1"), "pacifica": _pos("BTC", "ask", "1")},
    }
    bbos_by_symbol = {
        "BTC": {"omni": (D("100"), D("100.20")), "pacifica": (D("100.10"), D("100.30"))},
    }
    out = compute_spreads("increase", positions_by_symbol, bbos_by_symbol)
    # increase: long=omni ask 100.20, short=pacifica bid 100.10 → cost 0.10
    expected = (D("100.20") - D("100.10")) / ((D("100.20") + D("100.10")) / 2) * 100
    assert out["BTC"] == expected


def test_compute_spreads_returns_none_when_not_hedged_across_venues():
    positions_by_symbol = {
        "BTC": {"omni": _pos("BTC", "bid", "1"), "pacifica": _pos("BTC", "bid", "1")},
    }
    bbos_by_symbol = {"BTC": {"omni": (D("100"), D("101")), "pacifica": (D("100"), D("101"))}}
    out = compute_spreads("reduce", positions_by_symbol, bbos_by_symbol)
    assert out["BTC"] is None  # same side on both venues → not hedged


def test_compute_spreads_returns_none_when_position_missing():
    positions_by_symbol = {"BTC": {"omni": _pos("BTC", "bid", "1"), "pacifica": None}}
    bbos_by_symbol = {"BTC": {"omni": (D("100"), D("101")), "pacifica": (D("100"), D("101"))}}
    out = compute_spreads("increase", positions_by_symbol, bbos_by_symbol)
    assert out["BTC"] is None


def test_compute_spreads_returns_none_when_bbo_missing():
    positions_by_symbol = {
        "BTC": {"omni": _pos("BTC", "bid", "1"), "pacifica": _pos("BTC", "ask", "1")},
    }
    bbos_by_symbol = {"BTC": {"omni": (D("100"), D("101"))}}  # pacifica BBO missing
    out = compute_spreads("reduce", positions_by_symbol, bbos_by_symbol)
    assert out["BTC"] is None


# MARK: plan_resize


def _state():
    return {
        "omni": [_pos("BTC", "bid", "0.5"), _pos("ETH", "ask", "2.0")],
        "pacifica": [_pos("BTC", "ask", "0.5"), _pos("ETH", "bid", "2.0")],
    }


def _prices():
    return {
        "omni": {"BTC": D("100000"), "ETH": D("3000")},
        "pacifica": {"BTC": D("100000"), "ETH": D("3000")},
    }


def _lots():
    return {
        "omni": {"BTC": D("0.0001"), "ETH": D("0.001")},
        "pacifica": {"BTC": D("0.0001"), "ETH": D("0.001")},
    }


def test_plan_resize_reduce_uses_opposite_side_and_caps_at_position_size():
    plan = plan_resize(_state(), D("1000000"), "reduce", _prices(), _lots())
    by = {(leg.venue, leg.symbol): leg for leg in plan}

    # omni BTC is long (bid): reduce → sell (ask), reduce_only, qty <= position 0.5
    omni_btc = by[("omni", "BTC")]
    assert omni_btc.side == "ask"
    assert omni_btc.reduce_only is True
    assert omni_btc.qty <= D("0.5")
    assert omni_btc.pos_side == "bid"

    # pacifica BTC is short (ask): reduce → buy (bid)
    pac_btc = by[("pacifica", "BTC")]
    assert pac_btc.side == "bid"
    assert pac_btc.reduce_only is True
    assert pac_btc.qty <= D("0.5")
    assert pac_btc.pos_side == "ask"


def test_plan_resize_increase_uses_same_side_not_reduce_only():
    plan = plan_resize(_state(), D("500"), "increase", _prices(), _lots())
    by = {(leg.venue, leg.symbol): leg for leg in plan}

    # omni BTC is long (bid): increase → buy more (bid)
    omni_btc = by[("omni", "BTC")]
    assert omni_btc.side == "bid"
    assert omni_btc.reduce_only is False
    assert omni_btc.qty > 0
    assert omni_btc.pos_side == "bid"

    # omni ETH is short (ask): increase → sell more (ask)
    omni_eth = by[("omni", "ETH")]
    assert omni_eth.side == "ask"
    assert omni_eth.reduce_only is False
    assert omni_eth.pos_side == "ask"


def test_plan_resize_amounts_follow_weights():
    plan = plan_resize(_state(), D("500"), "increase", _prices(), _lots())
    usd_by_symbol: dict[str, Decimal] = {}
    for leg in plan:
        usd_by_symbol[leg.symbol] = usd_by_symbol.get(leg.symbol, D(0)) + leg.usd
    # per venue: BTC 300, ETH 200 (weights 1.5:1.0); summed across 2 venues
    assert usd_by_symbol["BTC"] == D("600.00")
    assert usd_by_symbol["ETH"] == D("400.00")


def test_plan_resize_qty_rounds_to_lot_size():
    plan = plan_resize(_state(), D("500"), "increase", _prices(), _lots())
    for leg in plan:
        if leg.qty > 0:
            lot = _lots()[leg.venue][leg.symbol]
            assert (leg.qty / lot) == (leg.qty / lot).quantize(D(1))


def test_plan_resize_reduce_caps_at_position_size_does_not_flip():
    # tiny amount that would round above position size after lot rounding
    plan = plan_resize(_state(), D("1e9"), "reduce", _prices(), _lots())
    for leg in plan:
        if leg.qty > 0:
            assert leg.reduce_only
            # qty must not exceed the open position size for that venue×symbol
            positions = {p.symbol: p for p in _state()[leg.venue]}
            assert leg.qty <= positions[leg.symbol].size


def test_plan_resize_missing_position_reduce_skips():
    state = {"omni": [_pos("BTC", "bid", "0.5")], "pacifica": [_pos("BTC", "ask", "0.5")]}
    # ETH has no open position anywhere
    plan = plan_resize(state, D("500"), "reduce", _prices(), _lots())
    eth_legs = [leg for leg in plan if leg.symbol == "ETH"]
    assert len(eth_legs) == 2  # both venues
    assert all(leg.qty == 0 for leg in eth_legs)
    assert all("no open position" in leg.note for leg in eth_legs)


def test_plan_resize_missing_position_increase_flags():
    state = {"omni": [_pos("BTC", "bid", "0.5")], "pacifica": [_pos("BTC", "ask", "0.5")]}
    plan = plan_resize(state, D("500"), "increase", _prices(), _lots())
    eth_legs = [leg for leg in plan if leg.symbol == "ETH"]
    assert all(leg.qty == 0 for leg in eth_legs)
    assert all("no open position to increase" in leg.note for leg in eth_legs)


def test_actionable_legs_filters_zero_qty():
    plan = plan_resize(_state(), D("500"), "increase", _prices(), _lots())
    assert len(actionable_legs(plan)) == 4  # 2 venues × 2 symbols, all open
    plan_with_skips = plan_resize(
        {"omni": [_pos("BTC", "bid", "0.5")], "pacifica": [_pos("BTC", "ask", "0.5")]},
        D("500"),
        "reduce",
        _prices(),
        _lots(),
    )
    # BTC actionable (2), ETH skipped (2) → 2 actionable
    assert len(actionable_legs(plan_with_skips)) == 2


# MARK: passive_farm account picker


def _acc(name: str, *, passive_farm=False, enabled=True) -> AccountConfig:
    return AccountConfig(
        name=name, privkey=SecretStr("0x" + "11" * 32), enabled=enabled, passive_farm=passive_farm
    )


def test_pick_passive_farm_returns_first_flagged_account():
    accs = [
        _acc("a", passive_farm=False),
        _acc("b", passive_farm=True),
        _acc("c", passive_farm=True),
    ]
    assert pick_passive_farm_account(accs, "omni").name == "b"


def test_pick_passive_farm_raises_when_none_flagged():
    accs = [_acc("a"), _acc("b")]
    with pytest.raises(ValueError, match="passive_farm"):
        pick_passive_farm_account(accs, "omni")


def test_pick_passive_farm_skips_disabled_even_if_flagged():
    accs = [
        _acc("disabled", passive_farm=True, enabled=False),
        _acc("active", passive_farm=True, enabled=True),
    ]
    assert pick_passive_farm_account(accs, "omni").name == "active"


def test_pick_passive_farm_raises_when_all_flagged_disabled():
    accs = [_acc("a", passive_farm=True, enabled=False)]
    with pytest.raises(ValueError, match="passive_farm"):
        pick_passive_farm_account(accs, "omni")


def test_pick_passive_farm_error_mentions_venue():
    with pytest.raises(ValueError, match="'omni'"):
        pick_passive_farm_account([], "omni")


# MARK: venue registry


def test_venue_registry_covers_all_known_venues():
    expected = {
        "ethereal",
        "hyena",
        "hyperliquid",
        "nado",
        "omni",
        "onyx",
        "pacifica",
        "rise",
        "zero1",
    }
    assert set(VENUE_REGISTRY) == expected
    assert sorted(VENUE_REGISTRY) == KNOWN_VENUES


def test_venue_registry_entries_have_client_and_config_path():
    for venue, (client_cls, cfg_path) in VENUE_REGISTRY.items():
        assert hasattr(client_cls, "from_config"), f"{venue} client missing from_config"
        assert cfg_path.startswith("configs/") and cfg_path.endswith(".toml"), venue
