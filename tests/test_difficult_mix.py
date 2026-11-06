"""Actual signal/gain invariants of the conditional foreground fallback."""
from copy import deepcopy
import numpy as np
import pytest
from superdj import difficult_mix as dm
from superdj.transition_quality import TransitionFailure, audit_vocals
from superdj.sync import shared_clock


def signals():
    sr = 8000; n = 32*sr; t = np.arange(n)/sr
    stereo = lambda x: np.column_stack([x, x]).astype("float32")
    bass = stereo(.07*np.sin(2*np.pi*80*t))
    bed = stereo(.025*np.sin(2*np.pi*600*t)+.02*np.sin(2*np.pi*4200*t))
    va = stereo(.05*np.sin(2*np.pi*900*t)*((t<17)|(t>18)))
    vb = stereo(.05*np.sin(2*np.pi*1100*t)*((t<22)|(t>23)))
    drums = stereo(.03*np.sin(2*np.pi*1800*t))
    return sr, bass+bed+va+drums, bass+bed+vb+drums, va, vb, drums


def rejection():
    return TransitionFailure("a", "b", "actual foreground refusal", {"vocals": {"passed": False}})


def test_one_continuous_voice_opening_and_native_quiet_exit():
    sr, a, b, va, vb, d = signals()
    recipe = dm.foreground_recipe(a,b,va,vb,sr,"outgoing")
    ga,gb = dm.voice_gains(len(a),sr,recipe)
    assert 17 <= recipe["outgoing_release_seconds"][0] < 18
    assert np.all(ga[:17*sr] == 1) and np.all(ga[18*sr:] == 0)
    assert gb[-1,0] == 1 and np.min(np.diff(gb[:,0])) >= 0
    main = round(recipe["incoming_db_points"][1][0]*sr)
    db = 20*np.log10(np.maximum(gb[main:,0],1e-12))
    assert db[0] == pytest.approx(-3) and db[-1] == 0
    assert np.max(db[sr//2:]-db[:-sr//2]) < .5
    # No dip or retrigger for each later word/voice activity change.
    assert len(recipe["incoming_db_points"]) == 3
    with pytest.raises(ValueError): dm.voice_gains(len(a)-1,sr,recipe)


def test_native_phrase_without_quiet_gap_is_refused_instead_of_word_cut():
    sr,a,b,va,vb,d = signals()
    va[:] = .03; vb[:] = .04
    with pytest.raises(ValueError,match="quiet release"):
        dm.foreground_recipe(a,b,va,vb,sr,"outgoing")


def test_outgoing_priority_preserves_strong_old_bed_while_next_is_backing():
    sr,a,b,va,vb,d = signals()
    recipe = dm.foreground_recipe(a,b,va,vb,sr,"outgoing")
    before = [x.copy() for x in (a,b,va,vb,d)]
    pa,pb = dm.prepare_foregrounds(a,b,va,vb,d,d,sr,recipe)
    # Estimated voices are unchanged/closed in this early, steady section.
    # The first deck keeps its bass/percussion and strong middle foundation.
    region = slice(sr,8*sr)
    np.testing.assert_allclose(pa[region],a[region],atol=2e-7)
    t = np.arange(len(a))/sr
    incoming_bed = (pb-d)[region, 0]
    project = lambda hz: 2*np.mean(incoming_bed*np.sin(2*np.pi*hz*t[region]))
    assert abs(20*np.log10(project(80)/.07)) < .02
    assert 20*np.log10(project(600)/.025) == pytest.approx(-12., abs=.1)
    assert np.array_equal(pa[0],a[0]) and np.array_equal(pb[-1],b[-1])
    for x,original in zip((a,b,va,vb,d),before): np.testing.assert_array_equal(x,original)


def test_energy_loss_is_not_hidden_by_quiet_second_source():
    sr=8000; t=np.arange(12*sr)/sr
    a=np.column_stack([.1*np.sin(2*np.pi*300*t)]*2).astype("float32")
    b=a*.01; y=a.copy(); y[4*sr:8*sr] *= .05
    assert dm.audit_impulse(a,b,y,sr)["passed"] is False
    # A genuine native break in BOTH sources is retained.
    a[4*sr:8*sr]=0; b[4*sr:8*sr]=0; y[4*sr:8*sr]=0
    assert dm.audit_impulse(a,b,y,sr)["passed"] is True


@pytest.mark.parametrize("control",[False,{"enabled":False},{"foreground_priority":"outgoing"}])
def test_settings_are_explicit_and_bounded(control):
    value=dm.settings(control)
    assert value["foreground_priority"] in {"auto","outgoing","incoming"}
    with pytest.raises(ValueError): dm.settings({"maximum_recipes":38})


@pytest.mark.parametrize("extra",[{"difficult_transition":False},{"handoff_windows":{"bass":[8,40]}},
    {"mid_handoff_bars":16},{"whole_spectrum_handoff":{}},{"mid_energy_mode":"legacy"}])
def test_refusal_never_overrides_explicit_handoff_or_disabled_policy(extra):
    assert not dm.eligible(rejection(),extra)
    assert not dm.eligible(ValueError("missing source file"),{})
    assert not dm.eligible(TransitionFailure("a","b","source clock",{"stage":"source-only timing preflight"}),{})


def test_actual_voice_audit_replays_new_gains_instead_of_legacy_fractions():
    sr,a,b,va,vb,d=signals();recipe=dm.foreground_recipe(a,b,va,vb,sr,"outgoing")
    mixed=np.ones_like(a)*.05
    transition={"beats":64,"vocal_processing":recipe}
    report=audit_vocals(a,b,mixed,sr,va,vb,transition,style="zhilin",beat_phase=None)
    assert report["passed"] and report["absence_certified"] is False
    invalid=deepcopy(transition);invalid["vocal_processing"]["frames"]-=1
    with pytest.raises(ValueError,match="different timed PCM"):
        audit_vocals(a,b,mixed,sr,va,vb,invalid,style="zhilin",beat_phase=None)


def setup_rescue(monkeypatch):
    from superdj import percussion as pm, transition_quality as tq
    sr,a,b,va,vb,d=signals();clock=shared_clock(120,120,64,sr)
    edge={"from":"a","to":"b","beats":64,"tempo_technique":"shared_phase_sync","shared_clock":clock}
    clips=[]
    for identity,role in (("a","tail"),("b","head")):
        clips.append({"id":identity,"warp":{"post_render_transient_audit":{"final":{"measured":[
            {"role":role,"beat":i,"delay_seconds":0.} for i in range(1,64)]}}}})
    calls=[]
    def estimates(*args,**kwargs):
        calls.append("estimates");assert args[0] is a and args[1] is b
        return {"drums_out":d,"drums_in":d},{"exact_final_decks":True}
    def fronts(audio,sr,wa,wb,g,directory,*,components,**kwargs):
        calls.append("fronts")
        np.testing.assert_allclose(components["tail"]+components["head"],audio,atol=2e-7)
        assert kwargs["allow_pitch_changing_residual"] is False
        return audio,{"executed":True,"checks":[{"role":role,"recognized":[{},{}]} for role in ("tail","head")]}
    monkeypatch.setattr(pm,"final_drum_estimates",estimates)
    monkeypatch.setattr(pm,"verify_blended_fronts",fronts)
    kwargs=dict(vocal_out=va,vocal_in=vb,initial_error=rejection(),beat_phase=np.linspace(0,1,len(a)),style="zhilin",ffmpeg="unused")
    return sr,a,b,clips,edge,calls,kwargs


def test_recovery_uses_exact_raw_decks_once_and_all_actual_quality_gates(tmp_path,monkeypatch):
    sr,a,b,clips,edge,calls,kw=setup_rescue(monkeypatch);before=deepcopy(edge)
    y,selected=dm.recover(a,b,sr,clips,edge,tmp_path,**kw)
    assert calls==["estimates","fronts"] and edge==before
    assert selected["quality_gate"]["passed"]
    assert selected["difficult_transition_recovery"]["raw_source_energy"]["passed"]
    assert selected["shared_clock"]==edge["shared_clock"] and len(y)==len(a)
    assert selected["difficult_transition_recovery"]["maximum_recipes"]==2


def test_failed_rhythm_remains_failed_and_does_not_repeat_models(tmp_path,monkeypatch):
    from superdj import percussion as pm
    sr,a,b,clips,edge,calls,kw=setup_rescue(monkeypatch)
    def failed(*args,**kwargs):calls.append("refused");raise ValueError("actual late front")
    monkeypatch.setattr(pm,"verify_blended_fronts",failed)
    error=kw["initial_error"]
    with pytest.raises(TransitionFailure) as caught: dm.recover(a,b,sr,clips,edge,tmp_path,**kw)
    assert caught.value is error and calls.count("estimates")==1 and calls.count("refused")<=2
    assert error.evidence["difficult_transition_recovery"]["passed"] is False


def test_missing_final_front_evidence_never_certifies_recovery(tmp_path,monkeypatch):
    from superdj import percussion as pm
    sr,a,b,clips,edge,calls,kw=setup_rescue(monkeypatch)
    monkeypatch.setattr(pm,"verify_blended_fronts",lambda pcm,*a,**k:(pcm,{"executed":True,"checks":[]}))
    with pytest.raises(TransitionFailure): dm.recover(a,b,sr,clips,edge,tmp_path,**kw)


def test_foreground_refusal_on_low_memory_does_not_launch_model(tmp_path,monkeypatch):
    from superdj import resources
    sr,a,b,clips,edge,calls,kw=setup_rescue(monkeypatch)
    monkeypatch.setattr(resources,"memory_policy",lambda:{"buffer_budget_bytes":1000})
    with pytest.raises(TransitionFailure): dm.recover(a,b,sr,clips,edge,tmp_path,**kw)
    assert not calls


def test_real_successful_sync_never_invokes_foreground_rescue_and_is_bit_exact(tmp_path,monkeypatch):
    from test_sync import RenderedSyncTests, FFMPEG, RUBBERBAND
    from superdj.render import render_plan
    from superdj.planner import plan_tracks
    import soundfile as sf
    if not FFMPEG or not RUBBERBAND: pytest.skip("Actual audio backends required")
    records=RenderedSyncTests().fixtures(tmp_path,[120,122])
    plan=plan_tracks(records,mode="fixed")
    def unexpected(*args,**kwargs):pytest.fail("Successful ordinary audio entered the reserve")
    monkeypatch.setattr(dm,"recover",unexpected)
    kwargs=dict(ffmpeg=FFMPEG,rubberband=RUBBERBAND,sample_rate=16000,
                tempo_conflict_policy="sync",sync_transition_bars=4)
    on=render_plan(plan,records,{},tmp_path/"on.wav",tmp_path/"on-work",**kwargs)
    off_plan=deepcopy(plan);off_plan.setdefault("mixing_preferences",{})["difficult_transition"]=False
    off=render_plan(off_plan,records,{},tmp_path/"off.wav",tmp_path/"off-work",**kwargs)
    np.testing.assert_array_equal(sf.read(tmp_path/"on.wav",dtype="float32")[0],sf.read(tmp_path/"off.wav",dtype="float32")[0])
    assert not on["transitions"][0].get("difficult_transition_recovery")
    assert on["transitions"][0]["quality_gate"]["impulse"]["passed"]
