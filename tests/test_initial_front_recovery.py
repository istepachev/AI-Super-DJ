from copy import deepcopy
from superdj.recovery import unsupported_attack_events


def evidence():
    return {'warp':{'source_kick_alignment':{'tail':{'events':[{'beat':4,'source_seconds':103.5},{'beat':5,'source_seconds':104.}],
        'native_mixture_reference':{'initial_front_required_beats':[4,5]}}}},
        'final_attack':{'measured':[{'role':'tail','beat':b,'view':view,'delay_seconds':0.}
            for b in (4,5)for view in ('primary','bass_body','estimated_bass')]}}


def test_missing_initial_front_cannot_be_retried_by_renumbering_a_source_beat():
    value=evidence();value['final_attack']['measured'].append({'role':'tail','beat':4,'view':'initial_front','delay_seconds':0.})
    before=deepcopy(value)
    assert unsupported_attack_events(value,'tail')==[104.]
    # Recovery stores physical source time, not a relative beat number.
    assert value==before
    value['final_attack']['measured'].append({'role':'tail','beat':5,'view':'initial_front','delay_seconds':.008})
    assert unsupported_attack_events(value,'tail')==[104.]


def test_recognized_initial_front_keeps_compatible_source_windows_available():
    value=evidence();value['final_attack']['measured'] += [{'role':'tail','beat':b,'view':'initial_front','delay_seconds':0.}for b in (4,5)]
    assert unsupported_attack_events(value,'tail')==[]
