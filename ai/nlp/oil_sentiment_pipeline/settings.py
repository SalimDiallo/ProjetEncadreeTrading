"""Chargement de config/pipeline.yaml + fusion avec les arguments CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML requis : pip install pyyaml") from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class PipelineSettings:
    sources: List[str] = field(default_factory=lambda: ["news", "reddit", "twitter", "edgar"])
    model: str = "auto"
    strategy: str = "threshold"
    freq: str = "1D"
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    ticker: str = "CL=F"
    buy_threshold: float = 0.05
    sell_threshold: float = -0.05
    smooth_window: int = 3
    max_per_source: int = 30
    transaction_cost: float = 0.001
    allow_short: bool = True
    save_data: bool = True
    plot: bool = True

    weight_by_confidence: bool = True
    weight_by_oil_density: bool = True
    weight_by_source: bool = True
    source_weights: Dict[str, float] = field(default_factory=dict)

    signal_lag: int = 1
    position_sizing: str = "binary"
    walk_forward_splits: int = 3

    allow_mock: bool = True
    min_text_length: int = 20
    filter_oil_keywords: bool = True

    gold_path: str = "tests/fixtures/sentiment_gold.csv"

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "PipelineSettings":
        raw = _load_yaml(path or DEFAULT_CONFIG_PATH)
        p = raw.get("pipeline", {})
        a = raw.get("aggregation", {})
        b = raw.get("backtest", {})
        ing = raw.get("ingestion", {})
        ev = raw.get("evaluation", {})
        return cls(
            sources=list(p.get("sources", cls().sources)),
            model=p.get("model", "auto"),
            strategy=p.get("strategy", "threshold"),
            freq=p.get("freq", "1D"),
            start_date=p.get("start_date", "2023-01-01"),
            end_date=p.get("end_date", "2024-12-31"),
            ticker=p.get("ticker", "CL=F"),
            buy_threshold=float(p.get("buy_threshold", 0.05)),
            sell_threshold=float(p.get("sell_threshold", -0.05)),
            smooth_window=int(p.get("smooth_window", 3)),
            max_per_source=int(p.get("max_per_source", 30)),
            transaction_cost=float(p.get("transaction_cost", 0.001)),
            allow_short=bool(p.get("allow_short", True)),
            save_data=bool(p.get("save_data", True)),
            plot=bool(p.get("plot", True)),
            weight_by_confidence=bool(a.get("weight_by_confidence", True)),
            weight_by_oil_density=bool(a.get("weight_by_oil_density", True)),
            weight_by_source=bool(a.get("weight_by_source", True)),
            source_weights=dict(a.get("source_weights", {})),
            signal_lag=int(b.get("signal_lag", 1)),
            position_sizing=b.get("position_sizing", "binary"),
            walk_forward_splits=int(b.get("walk_forward_splits", 3)),
            allow_mock=bool(ing.get("allow_mock", True)),
            min_text_length=int(ing.get("min_text_length", 20)),
            filter_oil_keywords=bool(ing.get("filter_oil_keywords", True)),
            gold_path=ev.get("gold_path", "tests/fixtures/sentiment_gold.csv"),
        )

    def merge_cli(self, **overrides) -> "PipelineSettings":
        """Retourne une copie avec les valeurs CLI non-None."""
        data = {**self.__dict__}
        for key, value in overrides.items():
            if value is not None and key in data:
                data[key] = value
        return PipelineSettings(**data)
