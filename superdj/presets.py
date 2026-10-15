"""Portable, explicitly selected experimental ordering profiles."""
from importlib.resources import files
import json


ORDERING_PRESETS = ('harmonic-route-v1', 'harmonic-route-v2', 'harmonic-next-v1')


def ordering_profile(name):
    """Load packaged inference parameters; never read a workstation profile."""
    if name not in ORDERING_PRESETS:
        raise ValueError('Unknown ordering preset')
    resource = files('superdj').joinpath('profiles', f'{name}.json')
    from .harmonize_modes import validate_mode_ranker_profile
    profile = validate_mode_ranker_profile(json.loads(resource.read_text(encoding='utf-8')))
    if profile.get('publication', {}).get('preset') != name:
        raise ValueError('Ordering preset identity differs')
    return profile
