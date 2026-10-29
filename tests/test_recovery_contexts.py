from copy import deepcopy
import hashlib
import pytest
from superdj.recovery_contexts import QualityContexts, merge_exclusions


def data():
    rows = [{'id': i, 'bpm': 124., 'provenance': {'source_sha256': hashlib.sha256(i.encode()).hexdigest()}}
        for i in ('a', 'b', 'c')]
    plan = {'style': 'zhilin', 'sections': {i: {'start': 20., 'end': 240.} for i in ('a', 'b', 'c')},
        'edges': [{'from': 'a', 'to': 'b', 'overlap_bars': 32, 'cue_locks': {'incoming_start_seconds': 20.}},
                  {'from': 'b', 'to': 'c', 'overlap_bars': 16, 'cue_locks': {'outgoing_end_seconds': 240.}}]}
    evidence = {'local_rhythm_coverage': {'passed': False, 'decks': [
        {'id': 'a', 'role': 'tail', 'sufficient_for_automatic_window': True},
        {'id': 'b', 'role': 'head', 'sufficient_for_automatic_window': False}]}}
    return plan, rows, evidence


def test_actual_local_refusal_survives_role_changes_without_becoming_a_source_ban():
    plan, rows, e = data(); q = QualityContexts(dependency_token='engine/prepared/private-inputs')
    assert q.note(plan, rows, e, 'local refusal') == 1
    e['local_rhythm_coverage']['decks'][1]['role'] = 'tail'
    assert q.note(plan, rows, e, 'same refusal') == 0
    assert q.exclusions(plan, rows, 'b') == {'b': [[20., 240.]]}
    assert not q.exclusions(plan, rows, 'a')
    assert q.document['native_recording_blacklist'] is False
    assert 'rejected_attack_window_events' not in plan


@pytest.mark.parametrize('change', [
    lambda p, r: r[1].update(rubberband_profile_override='short'),
    lambda p, r: r[1]['provenance'].update(source_sha256='1' * 64),
    lambda p, r: r[0]['provenance'].update(source_sha256='2' * 64),
    lambda p, r: r[2].update(rubberband_engine_override='r3'),
    lambda p, r: r[1].update(vocal_stem={'artifact_sha256': {'drums_path': 'new'}}),
    lambda p, r: p['sections']['a'].update(end=240.01),
    lambda p, r: p['edges'][0].update(overlap_bars=16),
    lambda p, r: p['edges'][0].update(mid_energy_mode='power'),
    lambda p, r: p['edges'][1].update(whole_spectrum_handoff={'start_beat': 4, 'end_beat': 36}),
    lambda p, r: p.update(style='sync'),
    lambda p, r: p.update(mixing_preferences={'profile': 'new'})])
def test_changed_processing_context_allows_fresh_checks(change):
    plan, rows, e = data(); q = QualityContexts(dependency_token='v1'); q.note(plan, rows, e, 'refusal')
    pp, rr = deepcopy(plan), deepcopy(rows); change(pp, rr)
    assert not q.exclusions(pp, rr, 'b')
    assert q.exclusions(plan, rows, 'b')


def test_persistence_reopens_after_dependency_change_and_does_not_alias_mutable_data():
    p, r, e = data(); q = QualityContexts(dependency_token='v1'); q.note(p, r, e, 'refusal')
    old = deepcopy(q.document); restored = QualityContexts(old, dependency_token='v1')
    old['items'].clear()
    exclusions = restored.exclusions(p, r, 'b'); exclusions['b'][0][0] = 999
    assert restored.exclusions(p, r, 'b') == {'b': [[20., 240.]]}
    assert not QualityContexts(q.document, dependency_token='new-preparation-or-engine').exclusions(p, r, 'b')


def test_own_cue_change_does_not_forget_old_complete_spans_or_round_new_spans_away():
    p, r, e = data(); q = QualityContexts(dependency_token='v1'); q.note(p, r, e, 'refusal')
    p['sections']['b']['start'] = 20.01; p['edges'][0]['cue_locks']['incoming_start_seconds'] = 20.01
    assert q.exclusions(p, r, 'b') == {'b': [[20., 240.]]}
    q.note(p, r, e, 'second exact span')
    assert q.exclusions(p, r, 'b') == {'b': [[20., 240.], [20.01, 240.]]}


@pytest.mark.parametrize('evidence', [{}, {'final_attack': {'passed': True}},
    {'local_rhythm_coverage': {'passed': True, 'decks': [{'id': 'b', 'sufficient_for_automatic_window': False}]}},
    {'local_rhythm_coverage': {'passed': False, 'decks': [{'id': 'b', 'sufficient_for_automatic_window': True}]}}])
def test_passed_unmeasured_and_unattributed_checks_do_not_create_exclusions(evidence):
    p, r, _ = data(); q = QualityContexts(dependency_token='v1')
    assert q.note(p, r, evidence, '') == 0 and not q.document['items']


def test_passing_deck_evidence_does_not_erase_an_actual_mix_refusal():
    p, r, e = data(); e['final_attack'] = {'passed': True}
    q = QualityContexts(dependency_token='v1')
    assert q.note(p, r, e, 'blend still failed') == 1


def test_missing_content_identity_cannot_create_a_persistent_exclusion():
    p, r, e = data(); r[0].pop('provenance'); q = QualityContexts(dependency_token='v1')
    assert q.note(p, r, e, 'unidentified neighbour') == 0


def test_bounded_history_and_independent_native_constraints_are_preserved():
    p, r, e = data(); q = QualityContexts(dependency_token='v1', maximum_entries=2)
    for start in (20., 21., 22.):
        p['sections']['b']['start'] = start; q.note(p, r, e, 'refusal')
    assert q.exclusions(p, r, 'b') == {'b': [[21., 240.], [22., 240.]]}
    assert merge_exclusions({'b': [[0., 1.]]}, q.exclusions(p, r, 'b'), {'b': [[0., 1.]]}) == {
        'b': [[0., 1.], [21., 240.], [22., 240.]]}
