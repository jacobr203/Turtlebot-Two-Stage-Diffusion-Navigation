"""Exceptions raised by the real-robot harness."""


class HarnessError(RuntimeError):
    """Infrastructure failure: the trial is re-run, never counted."""


class BatteryLow(RuntimeError):
    """Below the swap threshold: no trial starts, no row is written."""
