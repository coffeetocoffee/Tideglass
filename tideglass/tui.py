"""Terminal dashboard (TUI-lite) for Tideglass — v0.4 product surface.

A dependency-free, text-only dashboard that renders a ``predict``/``advise``
session: an ASCII sparkline of the height curve, the high/low, the rip-risk
band, harvesting windows, and species exposure. No curses/rich required, so it
runs anywhere and is easy to test (it just returns a string).

Use :func:`build_dashboard` for the string, or the ``tideglass tui`` CLI for the
live view.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from tideglass.marea.model import TideModel
from tideglass.marine.advisor import TideAdvisor

# ASCII ramp (portable: renders on any codepage, unlike Unicode block chars).
_SPARK = " .:-=+*#%@"


def _sparkline(values: np.ndarray, width: int = 48) -> str:
    if values.size == 0:
        return ""
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo or 1.0
    n = values.size
    step = max(1, n // width)
    chars = []
    for i in range(0, n, step):
        v = values[i]
        idx = int((v - lo) / span * (len(_SPARK) - 1))
        chars.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(chars)


def build_dashboard(
    model: TideModel,
    times: Sequence[datetime],
    advisor: TideAdvisor | None = None,
) -> str:
    """Render a text dashboard for a fitted model over ``times``."""
    times = list(times)
    pred = model.predict(times)
    mean = pred.mean
    hi = float(mean.max())
    lo = float(mean.min())
    peak_i = int(mean.argmax())
    low_i = int(mean.argmin())
    lines = [
        f"Tideglass - station {model.station or '?'}   ({times[0]:%Y-%m-%d %H:%M} -> {times[-1]:%Y-%m-%d %H:%M})",
        "",
        f"  high {hi:6.2f} m @ {times[peak_i]:%m-%d %H:%M}   low {lo:6.2f} m @ {times[low_i]:%m-%d %H:%M}",
        f"  range {hi - lo:6.2f} m",
        "",
        "  height curve:",
        "  " + _sparkline(mean),
        "",
    ]
    if advisor is None:
        advisor = TideAdvisor(model)
    advice = advisor.advise(times)
    rip_peak = int(advice.rip.score.argmax())
    lines.append(f"  rip risk: {advice.rip.category[rip_peak]} "
                 f"({advice.rip.score[rip_peak]:.2f}) @ {advice.rip.times[rip_peak]:%m-%d %H:%M}")
    lines.append(f"  harvest windows: {len(advice.harvest)}")
    for w in advice.harvest[:5]:
        lines.append(f"    {w.start:%m-%d %H:%M}->{w.end:%H:%M}  ({w.reason})")
    lines.append("  species exposure (fraction of period):")
    for name, frac in advice.exposure.items():
        lines.append(f"    {name:<14} {frac * 100:5.0f}%")
    return "\n".join(lines)
