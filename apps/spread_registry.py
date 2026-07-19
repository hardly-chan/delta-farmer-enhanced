# delta-farmer | https://github.com/vladkens/delta-farmer
# Copyright (c) vladkens | MIT License | Venue registry for the spread resizer
"""Maps each supported venue name to its trading client class and default config path.

Kept in its own module so `strategy/spread.py` (the engine) stays free of app-layer
imports, and so `apps/spread.py` has a single place to resolve venues.
"""

# imports are explicit so an unknown/broken client fails loudly at import time
from apps.hyperliquid import HyperLiquidNativeClient
from clients.ethereal import EtherealClient
from clients.hyena import HyenaClient
from clients.nado import NadoClient
from clients.omni import OmniClient
from clients.onyx import OnyxClient
from clients.pacifica import PacificaClient
from clients.rise import RiseClient
from clients.zero1 import ZeroOneClient
from strategy.models import TradingClient

# venue name -> (client class, default config path relative to project root)
VENUE_REGISTRY: dict[str, tuple[type[TradingClient], str]] = {
    "ethereal": (EtherealClient, "configs/ethereal.toml"),
    "hyena": (HyenaClient, "configs/hyena.toml"),
    "hyperliquid": (HyperLiquidNativeClient, "configs/hyperliquid.toml"),
    "nado": (NadoClient, "configs/nado.toml"),
    "omni": (OmniClient, "configs/omni.toml"),
    "onyx": (OnyxClient, "configs/onyx.toml"),
    "pacifica": (PacificaClient, "configs/pacifica.toml"),
    "rise": (RiseClient, "configs/rise.toml"),
    "zero1": (ZeroOneClient, "configs/zero1.toml"),
}

KNOWN_VENUES: list[str] = sorted(VENUE_REGISTRY)
