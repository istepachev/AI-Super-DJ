"""Observable safety-policy behavior and preserved musical constraints."""
import copy
import pytest

from superdj.preset_routes import sequence_ordering_preset
from superdj.presets import ordering_profile
from superdj.harmonize_modes_sequence import sequence_mode_tracks


def records():
    return [dict(id=str(i), bpm=118+i/2, key='8A') for i in range(20)]


def test_same_musical_safety_retains_primary_and_explicit_constraints():
    rows = records()
    arguments = dict(mode='mood', priority=0, routing_priority='bpm', bpm_tolerance=4,
                     direction='none', seed=0, first_id='0', last_id='19', locked_chains=[['8','9']])
    result = sequence_ordering_preset(list(reversed(rows)), 'djstudio-route-v2', **arguments)
    primary = sequence_mode_tracks(sorted(rows, key=lambda row:row['id']),
                                   ordering_profile('djstudio-route-v2'), **arguments)
    assert result['order'] == primary['order']
    assert len(result['order']) == len(set(result['order'])) == 20
    assert result['order'][0] == '0' and result['order'][-1] == '19'
    assert result['order'][result['order'].index('8')+1] == '9'
    decision = result['preset_route_selection']
    assert decision['selected'] == 'primary'
    assert decision['primary_outside_mode_edges'] == decision['safety_outside_mode_edges'] == 0
    assert decision['teacher_order_used'] is decision['cross_model_utility_compared'] is False


def test_safety_selection_uses_harmony_even_when_primary_utility_is_larger(monkeypatch):
    # Control the backend's completed reports, including utility scales that
    # cannot be compared between models. The wrapper receives no teacher.
    import superdj.preset_routes as module
    rows = records()
    ids = [str(i) for i in range(20)]
    primary = dict(order=ids, edges=[{} for _ in range(19)], harmonic_matches=18,
                   model_utility=1e6, constraints={'all_tracks_once':True})
    safety = dict(order=list(reversed(ids)), edges=[{} for _ in range(19)], harmonic_matches=19,
                  model_utility=-1e6, constraints={'all_tracks_once':True})
    calls = []
    def backend(inputs, profile, **settings):
        calls.append((copy.deepcopy(inputs), settings))
        return copy.deepcopy(primary if len(calls) == 1 else safety)
    monkeypatch.setattr(module, 'sequence_mode_tracks', backend)
    result = sequence_ordering_preset(rows, 'djstudio-route-v2', mode='fuzzy', direction='none', seed=0)
    assert result['order'] == safety['order']
    assert result['preset_route_selection']['selected'] == 'safety'
    assert calls[0] == calls[1]
    assert result['preset_route_selection']['candidates'][0]['order'] == primary['order']


def test_bad_candidate_membership_cannot_be_silently_selected(monkeypatch):
    import superdj.preset_routes as module
    monkeypatch.setattr(module, 'sequence_mode_tracks', lambda *args, **kw:
        dict(order=['missing'], edges=[], harmonic_matches=0))
    with pytest.raises(RuntimeError, match='occurrence membership'):
        sequence_ordering_preset(records(), 'djstudio-route-v2')
