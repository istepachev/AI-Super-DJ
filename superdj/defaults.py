"""Portable ordinary-set preferences, with explicit choices taking precedence."""
from __future__ import annotations

STYLE_CHOICES = ("zhilin", "sync", "default")
HARMONIC_MODES = ("mood", "fuzzy", "classic")
ENERGY_ARCS = ("build", "release", "wave", "flat", "none")


def playback_defaults(*, long_play=False, target_play_seconds=None):
    """Long tracks require a per-request opt-in, never workstation state."""
    import math
    target = float(target_play_seconds) if target_play_seconds is not None else 360. if long_play else 210.
    if not math.isfinite(target) or target < 180 or (not long_play and target > 240):
        raise ValueError('CUE chapter target must be 180–240 seconds; longer targets require --long-play')
    return {'target_play_seconds': target, 'maximum_play_seconds': max(300., target+60.) if long_play else 300.,
            'target_play_range_seconds': [target-30., target+30.] if long_play else [180.,240.],
            'allow_long_play': bool(long_play), 'duration_budget_basis': 'chapter interval',
            'chapter_dominance': {'minimum_fraction': .8, 'preferred_fraction': .9}}


def planning_defaults(*, style=None, harmonic_mode=None, energy_arc=None, runtime=None):
    """Resolve CLI > private workstation setting > portable product default.

    ``default`` remains the original ordinary-SYNC style alias. Zhilin uses
    Mood unless explicitly overridden; other styles retain Classic. The
    musical arc is selected independently.
    """
    runtime = {} if runtime is None else runtime
    resolved = {}
    sources = {}
    for name, explicit, config_key, portable, choices in (
        ("style", style, "default_style", "zhilin", STYLE_CHOICES),
        ("harmonic_mode", harmonic_mode, "default_harmonic_mode", "classic", HARMONIC_MODES),
        ("energy_arc", energy_arc, "default_energy_arc", "build", ENERGY_ARCS),
    ):
        if name == "harmonic_mode" and resolved["style"] == "zhilin":
            portable = "mood"
        configured = runtime.get(config_key)
        value = explicit if explicit is not None else configured if configured is not None else portable
        if value not in choices:
            raise ValueError(f"{config_key} / explicit {name} must be one of {', '.join(choices)}")
        resolved[name] = "default" if name == "style" and value == "sync" else value
        sources[name] = "explicit" if explicit is not None else config_key if configured is not None else "portable default"
    return {**resolved, "preference_sources": sources}
