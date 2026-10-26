"""Define every live and archived signal block in one authoritative registry.

Reads no files and writes nothing.  Production composite construction,
research evaluation, and the stock explainer all resolve builders, constituent
signals, labels, and required panel columns here.  A new block belongs in one
registry entry; archived builders are lazy so rejected research code is not
loaded on the live portfolio path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from src.signals.capital_deployment import (
    build_capital_deployment_block,
    share_chg_3m,
    share_chg_12m,
    share_chg_36m,
)
from src.signals.momentum import build_momentum_block, momentum_12_1
from src.signals.quality import (
    asset_turnover_vol,
    build_quality_block,
    cfo_to_ic,
    ebit_margin_vol,
    ebitda_roic,
    eps_vol,
    roe_ltm,
)
from src.signals.sentiment import build_sentiment_block, short_interest_ratio
from src.signals.value import (
    build_value_block,
    cfo_yield,
    earnings_yield,
    ebitda_to_ev,
)


SignalFunction = Callable[[pd.DataFrame], pd.Series]
BlockBuilder = Callable[[pd.DataFrame], Any]


@dataclass(frozen=True)
class SignalSpec:
    """Metadata and callable for one constituent signal."""

    name: str
    label: str
    function: SignalFunction
    display_as_percent: bool = False


@dataclass(frozen=True)
class SignalBlockSpec:
    """Single source of truth for one signal block and its data contract."""

    name: str
    label: str
    builder: BlockBuilder
    block_attribute: str
    signals: tuple[SignalSpec, ...]
    panel_columns: tuple[str, ...]
    live: bool
    archived: bool = False

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Return constituent signal names in stable evaluation order."""
        return tuple(signal.name for signal in self.signals)

    def build(self, panel: pd.DataFrame) -> Any:
        """Run this block's registered builder on a point-in-time panel."""
        return self.builder(panel)

    def score(self, result: Any) -> pd.Series:
        """Extract the standardized block score from a builder result."""
        return getattr(result, self.block_attribute)


def _build_archived_risk(panel: pd.DataFrame) -> Any:
    """Load and build the rejected risk block only when research requests it."""
    from src.signals.archive.risk import build_risk_block

    return build_risk_block(panel)


def _archived_return_volatility(panel: pd.DataFrame) -> pd.Series:
    """Return the archived low-return-volatility signal."""
    from src.signals.archive.risk import return_volatility

    return return_volatility(panel)


def _archived_market_beta(panel: pd.DataFrame) -> pd.Series:
    """Return the archived low-beta signal, including its research sign flip."""
    from src.signals.archive.risk import market_beta

    return market_beta(panel)


def _archived_idio_volatility(panel: pd.DataFrame) -> pd.Series:
    """Return the archived low-idiosyncratic-volatility signal."""
    from src.signals.archive.risk import idio_volatility

    return idio_volatility(panel)


def _build_archived_fundamental_momentum(panel: pd.DataFrame) -> Any:
    """Load and build rejected fundamental momentum only for research."""
    from src.signals.archive.fundamental_momentum import (
        build_fundamental_momentum_block,
    )

    return build_fundamental_momentum_block(panel)


def _archived_cfo_trend(panel: pd.DataFrame) -> pd.Series:
    """Return the archived scaled quarterly CFO trend."""
    from src.signals.archive.fundamental_momentum import cfo_trend

    return cfo_trend(panel)


def _archived_revenue_trend(panel: pd.DataFrame) -> pd.Series:
    """Return the archived scaled quarterly revenue trend."""
    from src.signals.archive.fundamental_momentum import revenue_trend

    return revenue_trend(panel)


_STANDARDIZATION_COLUMNS = ("date", "investable", "gsector")
_QUARTERLY_KEYS = ("gvkey", "datadate", "available_date", "fyearq", "fqtr")


BLOCK_REGISTRY: Mapping[str, SignalBlockSpec] = {
    "value": SignalBlockSpec(
        name="value",
        label="Value",
        builder=build_value_block,
        block_attribute="value_block",
        signals=(
            SignalSpec("cfo_yield", "CFO yield", cfo_yield, True),
            SignalSpec("earnings_yield", "Earnings yield", earnings_yield, True),
            SignalSpec("ebitda_to_ev", "EBITDA / EV", ebitda_to_ev, True),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS
        + _QUARTERLY_KEYS
        + ("oancf_qtr", "ibq", "oibdpq", "dlcq", "dlttq", "cheq", "market_cap"),
        live=True,
    ),
    "quality": SignalBlockSpec(
        name="quality",
        label="Quality",
        builder=build_quality_block,
        block_attribute="quality_block",
        signals=(
            SignalSpec("cfo_to_ic", "CFO / invested capital", cfo_to_ic, True),
            SignalSpec("ebitda_roic", "EBITDA ROIC", ebitda_roic, True),
            SignalSpec("roe_ltm", "ROE, trailing 12 months", roe_ltm, True),
            SignalSpec(
                "asset_turnover_vol",
                "Asset-turnover stability",
                asset_turnover_vol,
                True,
            ),
            SignalSpec(
                "ebit_margin_vol",
                "EBIT-margin stability",
                ebit_margin_vol,
                True,
            ),
            SignalSpec("eps_vol", "EPS stability", eps_vol),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS
        + _QUARTERLY_KEYS
        + (
            "oancf_qtr",
            "oibdpq",
            "ibq",
            "saleq",
            "oiadpq",
            "epspxq",
            "ceqq",
            "atq",
            "dlcq",
            "dlttq",
            "cheq",
        ),
        live=True,
    ),
    "capital_deployment": SignalBlockSpec(
        name="capital_deployment",
        label="Capital deployment",
        builder=build_capital_deployment_block,
        block_attribute="capital_deployment_block",
        signals=(
            SignalSpec("share_chg_3m", "Share-count change, 3 months", share_chg_3m, True),
            SignalSpec(
                "share_chg_12m",
                "Share-count change, 12 months",
                share_chg_12m,
                True,
            ),
            SignalSpec(
                "share_chg_36m",
                "Share-count change, 36 months",
                share_chg_36m,
                True,
            ),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS
        + ("permno", "shares_out", "mthcumfacshr"),
        live=True,
    ),
    "momentum": SignalBlockSpec(
        name="momentum",
        label="Momentum",
        builder=build_momentum_block,
        block_attribute="momentum_block",
        signals=(
            SignalSpec("momentum_12_1", "Price momentum, 12-1", momentum_12_1, True),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS + ("permno", "ret"),
        live=True,
    ),
    "sentiment": SignalBlockSpec(
        name="sentiment",
        label="Sentiment",
        builder=build_sentiment_block,
        block_attribute="sentiment_block",
        signals=(
            SignalSpec(
                "short_interest_ratio",
                "Short-interest ratio",
                short_interest_ratio,
                True,
            ),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS
        + ("short_interest_adjusted_shares", "shares_out", "mthcumfacshr"),
        live=True,
    ),
    "risk": SignalBlockSpec(
        name="risk",
        label="Risk (archived)",
        builder=_build_archived_risk,
        block_attribute="risk_block",
        signals=(
            SignalSpec(
                "return_volatility",
                "Low return volatility",
                _archived_return_volatility,
                True,
            ),
            SignalSpec("market_beta", "Low market beta", _archived_market_beta),
            SignalSpec(
                "idio_volatility",
                "Low idiosyncratic volatility",
                _archived_idio_volatility,
                True,
            ),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS + ("permno", "ret"),
        live=False,
        archived=True,
    ),
    "fundamental_momentum": SignalBlockSpec(
        name="fundamental_momentum",
        label="Fundamental momentum (archived)",
        builder=_build_archived_fundamental_momentum,
        block_attribute="fundamental_momentum_block",
        signals=(
            SignalSpec("cfo_trend", "Quarterly CFO trend", _archived_cfo_trend, True),
            SignalSpec(
                "revenue_trend",
                "Quarterly revenue trend",
                _archived_revenue_trend,
                True,
            ),
        ),
        panel_columns=_STANDARDIZATION_COLUMNS
        + _QUARTERLY_KEYS
        + ("oancf_qtr", "revtq"),
        live=False,
        archived=True,
    ),
}


def get_block_spec(name: str) -> SignalBlockSpec:
    """Return a registered block or raise a clear error for an unknown name."""
    try:
        return BLOCK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown signal block {name!r}; registered blocks are "
            f"{list(BLOCK_REGISTRY)}."
        ) from exc


def validate_live_blocks(names: Iterable[str]) -> tuple[SignalBlockSpec, ...]:
    """Resolve configured blocks and require each to be live and unique."""
    ordered_names = tuple(names)
    if not ordered_names:
        raise ValueError("At least one live signal block is required.")
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("Configured signal blocks contain duplicate entries.")
    specs = tuple(get_block_spec(name) for name in ordered_names)
    invalid = [spec.name for spec in specs if not spec.live or spec.archived]
    if invalid:
        raise ValueError(f"Archived/non-live blocks cannot enter the model: {invalid}")
    return specs


def required_panel_columns(
    names: Iterable[str],
    base: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return a stable de-duplicated union of registered panel requirements."""
    columns = list(base)
    for name in names:
        columns.extend(get_block_spec(name).panel_columns)
    return tuple(dict.fromkeys(columns))


__all__ = [
    "BLOCK_REGISTRY",
    "SignalBlockSpec",
    "SignalSpec",
    "get_block_spec",
    "required_panel_columns",
    "validate_live_blocks",
]
