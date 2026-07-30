"""Registry helpers for the reference advisory bots."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Type

from ..signal import SignalPluginRegistry
from .base import AdvisoryBot
from .samples import (
    CatalystRiskConfig,
    CatalystRiskMonitor,
    DerivativesPositioningConfig,
    DerivativesPositioningMonitor,
    ExchangeEventConfig,
    ExchangeEventMonitor,
    MacroInstitutionalConfig,
    MacroInstitutionalMonitor,
    PriceVolumeConfig,
    PriceVolumeMonitor,
    SocialSentimentConfig,
    SocialSentimentMonitor,
)


_BOT_TYPES: Dict[str, Type[AdvisoryBot]] = {
    "price_volume_monitor": PriceVolumeMonitor,
    "social_sentiment_monitor": SocialSentimentMonitor,
    "exchange_event_monitor": ExchangeEventMonitor,
    "macro_institutional_monitor": MacroInstitutionalMonitor,
    "catalyst_risk_monitor": CatalystRiskMonitor,
    "derivatives_positioning_monitor": DerivativesPositioningMonitor,
}

_CONFIG_TYPES: Dict[str, Any] = {
    "price_volume_monitor": PriceVolumeConfig,
    "social_sentiment_monitor": SocialSentimentConfig,
    "exchange_event_monitor": ExchangeEventConfig,
    "macro_institutional_monitor": MacroInstitutionalConfig,
    "catalyst_risk_monitor": CatalystRiskConfig,
    "derivatives_positioning_monitor": DerivativesPositioningConfig,
}


def create_advisory_bot(bot_id: str) -> AdvisoryBot:
    try:
        return _BOT_TYPES[bot_id]()
    except KeyError:
        raise KeyError("unknown advisory bot %r" % bot_id)


def list_advisory_bots() -> List[Dict[str, Any]]:
    return [create_advisory_bot(bot_id).meta.to_dict() for bot_id in sorted(_BOT_TYPES)]


def register_advisory_bots(registry: SignalPluginRegistry) -> SignalPluginRegistry:
    for bot_id in _BOT_TYPES:
        bot = create_advisory_bot(bot_id)
        config_type = _CONFIG_TYPES[bot_id]
        registry.register(bot, config_type.from_dict)
    return registry


def canvas_definition(bot_id: str, config: Mapping[str, Any] = None) -> Dict[str, Any]:
    bot = create_advisory_bot(bot_id)
    config_type = _CONFIG_TYPES[bot_id]
    normalized = config_type.from_dict(config)
    return bot.to_canvas_dsl(asdict(normalized))
