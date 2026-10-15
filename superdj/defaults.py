"""Portable ordinary-set preferences, with explicit choices taking precedence."""
from __future__ import annotations

STYLE_CHOICES = ("zhilin", "sync", "default")
HARMONIC_MODES = ("mood", "fuzzy", "classic")
ENERGY_ARCS = ("build", "release", "wave", "flat", "none")


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
