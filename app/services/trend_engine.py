"""
TrendEngine — calculates trend from observation history.
Kept separate from AI inference and decision engine.
"""
from __future__ import annotations

from typing import Optional


class TrendEngine:
    """
    Computes trend (IMPROVING | STABLE | INCREASING) from a list of severity values.
    Input: list of severity values ordered oldest-first.
    """

    DELTA_THRESHOLD = 5.0  # minimum change to register a trend

    @classmethod
    def compute(cls, severity_history: list[float]) -> str:
        """Returns trend string from historical severity values."""
        if len(severity_history) < 2:
            return "STABLE"
        recent = severity_history[-3:] if len(severity_history) > 3 else severity_history
        delta = recent[-1] - recent[0]
        if delta > cls.DELTA_THRESHOLD:
            return "INCREASING"
        if delta < -cls.DELTA_THRESHOLD:
            return "IMPROVING"
        return "STABLE"
