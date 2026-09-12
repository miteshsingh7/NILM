"""Non-Intrusive Load Monitoring (NILM) multi-target seq2seq package."""

__version__ = "0.1.0"

from src.event_detection import (
    detect_appliance_state,
    evaluate_appliance_events,
    evaluate_disaggregation_events,
    format_metrics_table,
)

__all__ = [
    "detect_appliance_state",
    "evaluate_appliance_events",
    "evaluate_disaggregation_events",
    "format_metrics_table",
]
