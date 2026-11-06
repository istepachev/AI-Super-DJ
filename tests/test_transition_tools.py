"""Conditional timeline edits preserve actual attacks and unaffected PCM."""
from copy import deepcopy
import os
import numpy as np
import pytest
import soundfile as sf
from superdj import transition_tools as tools
from superdj.transition_quality import TransitionFailure


def refusal():
    return TransitionFailure("a","b","listener rejected overlap",{"listener_review":{"passed":False}})


def riff():
    sr=8000;t=np.arange(14*sr)/sr
    d=np.column_stack([.06*np.sin(2*np.pi*80*t)+.02*np.sin(2*np.pi*400*t)]*2).astype("float32")
    streams={"mixture":d*2,"voice":np.zeros_like(d),"drums":d}
    return sr,streams,dict(first_frame=4*sr,last_frame=8*sr,join_frames=sr//2,copies=2,
        beat_times=np.arange(0,14,.5),phase_origin_seconds=4.,failure=refusal())


def test_riff_copies_real_attacks_without_rescaling_and_all_streams_stay_aligned():
    sr,streams,kw=riff();kw["beat_times"][10]+=.003
    before={k:v.copy() for k,v in streams.items()}
    result,beats,report=tools.extend_riff(streams,sr,**kw)
    assert {x.shape for x in result.values()}=={(22*sr,2)}
    assert 9.003 in beats and 13.003 in beats
    np.testing.assert_allclose(result["mixture"],2*result["drums"],atol=2e-8)
    for key in streams:
        np.testing.assert_array_equal(result[key][16*sr:],streams[key][8*sr:])
        np.testing.assert_array_equal(streams[key],before[key])
    assert report["candidate_only"] and report["native_attack_offsets_preserved"]


def test_splice_boundary_is_not_a_fake_phase_origin():
    sr,streams,kw=riff();kw["first_frame"]+=round(.08*sr);kw["last_frame"]+=round(.08*sr)
    # The beat at 4.0 precedes the cut at 4.08. It is in the preroll, not a
    # reason to slide every copied drum onto a uniformly rescaled grid.
    result,beats,report=tools.extend_riff(streams,sr,**kw)
    assert report["phase_origin_seconds"]==4. and 8.5 in beats and 12.5 in beats
    assert not np.any(np.isclose(beats,8.08))


@pytest.mark.parametrize("change",[{"copies":38},{"phase_origin_seconds":3.},{"protected_ranges":[[1,2]]}])
def test_unsafe_loop_is_refused(change):
    sr,streams,kw=riff();kw.update(change)
    with pytest.raises(ValueError):tools.extend_riff(streams,sr,**kw)


def test_uneven_native_seam_or_foreground_loop_is_refused():
    sr,streams,kw=riff();kw["beat_times"][10]+=.1
    with pytest.raises(ValueError,match="five-percent"):tools.extend_riff(streams,sr,**kw)
    sr,streams,kw=riff();streams["voice"]=streams["mixture"]*.5
    with pytest.raises(ValueError,match="instrumental riff"):tools.extend_riff(streams,sr,**kw)


def test_patch_has_soft_margins_and_bit_exact_unaffected_audio():
    rng=np.random.default_rng(42);mixed=rng.normal(0,.1,(80000,2)).astype("float32")
    old=mixed*.3;new=old*.4
    result,proof=tools.patch_contribution(mixed,old,new,8000,first_seconds=3,last_seconds=7,failure=refusal())
    np.testing.assert_array_equal(result[:24000],mixed[:24000]);np.testing.assert_array_equal(result[56000:],mixed[56000:])
    np.testing.assert_array_equal(result[24000],mixed[24000]);np.testing.assert_array_equal(result[55999],mixed[55999])
    assert not np.array_equal(result[32000:48000],mixed[32000:48000])
    assert proof["candidate_only"]


def test_tools_never_run_on_success_or_unspecified_failure():
    sr,streams,kw=riff();kw["failure"]=ValueError("unknown")
    with pytest.raises(ValueError,match="actual quality refusal"):tools.extend_riff(streams,sr,**kw)


def prepared(tmp_path):
    sr=8000;t=np.arange(sr*3)/sr;result={"provenance":{}}
    for key,f in (("path",440.),("vocal_path",660.),("drum_path",110.)):
        path=tmp_path/(key+".wav");sf.write(path,np.column_stack([.06*np.sin(2*np.pi*f*t)]*2),sr,subtype="FLOAT")
        result[key]=str(path)
    return result


def test_pitch_is_whole_source_before_timing_and_never_combined_with_timemap(tmp_path,monkeypatch):
    from superdj import render
    item=prepared(tmp_path);original=deepcopy(item);commands=[]
    def run(args,**kwargs):
        commands.append(args)
        assert "--timemap" not in args and args[args.index("--time")+1]=="1"
        audio,sr=sf.read(args[-2],dtype="float32");sf.write(args[-1],audio,sr,subtype="FLOAT")
    monkeypatch.setattr(render,"_run",run)
    result=tools.pitch_preparation(item,tmp_path/"pitch",semitones=1,rubberband="backend",failure=refusal())
    assert len(commands)==3 and item==original
    assert result["provenance"]["pitch_adjustment"]["scope"].startswith("entire source")
    with pytest.raises(ValueError,match="already assigned"):
        tools.pitch_preparation(result,tmp_path/"second",semitones=-1,rubberband="backend",failure=refusal())


def test_real_pitch_stage_changes_frequency_and_preserves_full_timeline(tmp_path):
    backend=os.environ.get("SUPERDJ_RUBBERBAND")
    if not backend:pytest.skip("Optional real pitch executable")
    item=prepared(tmp_path)
    result=tools.pitch_preparation(item,tmp_path/"real-pitch",semitones=1,rubberband=backend,failure=refusal())
    audio,sr=sf.read(result["path"]);mono=audio[sr:-sr,0]
    f=np.fft.rfftfreq(len(mono),1/sr)[np.argmax(abs(np.fft.rfft(mono*np.hanning(len(mono)))))]
    assert f==pytest.approx(440*2**(1/12),abs=2.)
    assert sf.info(result["path"]).frames==sf.info(item["path"]).frames


def test_longer_overlap_preserves_automatic_neighbours_and_user_locks(tmp_path,monkeypatch):
    from superdj import compact
    plan={"order":["a","b","c"],"style":"zhilin","edges":[{"from":"a","to":"b","overlap_bars":8,
        "cue_locks":{"incoming_start_seconds":8.,"outgoing_end_seconds":200.}}, {"from":"b","to":"c","overlap_bars":8}],
        "sections":{k:{"start":8.,"end":200.,"locked_start":False,"locked_end":False} for k in "abc"}}
    records=[{"id":k} for k in "abc"];before=deepcopy(plan)
    def arrange(proposal,rows,**kwargs):
        assert proposal["edges"][0]["overlap_bars"]==16 and not proposal["edges"][0]["cue_locks"]
        assert kwargs["section_ranges"]["c"]=={"start":[8.,8.],"end":[200.,200.]}
        return {**proposal,"sections":deepcopy(before["sections"])}
    monkeypatch.setattr(compact,"arrange_sync",arrange)
    result=tools.longer_overlap(plan,records,refusal(),bars=16)
    assert result["order"]==plan["order"] and plan==before
    with pytest.raises(ValueError,match="longer"):tools.longer_overlap(plan,records,refusal(),bars=4)
    plan["sections"]["a"]["locked_end"]=True
    with pytest.raises(ValueError,match="locked/protected"):tools.longer_overlap(plan,records,refusal(),bars=16)


def test_shifted_source_cannot_keep_native_drum_template_paths(tmp_path,monkeypatch):
    item=prepared(tmp_path);item["provenance"]["pitch_adjustment"]={"semitones":1}
    row={"id":"a","kick_evidence":{"head":{"path":"old.wav","source_start_seconds":8.}},
         "vocal_stem":{"path":"old-voice.wav","drums_path":"old-drums.wav"}}
    original=deepcopy(row)
    result=tools.retarget_pitch_record(row,item)
    assert row==original and result["vocal_stem"]["path"]==item["vocal_path"]
    assert all(v["path"]==item["drum_path"] and v["source_start_seconds"]==0 for v in result["kick_evidence"].values())


def test_pitch_tools_refuse_external_storage_before_any_output(tmp_path,monkeypatch):
    from superdj import storage_budget
    allowed=tmp_path/"allowed";outside=tmp_path/"outside"
    monkeypatch.setattr(storage_budget,"workstation_policy",lambda:{"storage_root":str(allowed),"storage_max_gb":64})
    with pytest.raises(storage_budget.StorageLimitError,match="configured working folder"):
        tools.pitch_preparation({},outside,semitones=1,rubberband="must-not-run",failure=refusal())
    assert not outside.exists() and not allowed.exists()


def test_pitch_tools_count_the_shared_working_limit_before_output(tmp_path,monkeypatch):
    from superdj import storage_budget
    item=prepared(tmp_path);target=tmp_path/"pitch-budget"
    monkeypatch.setattr(storage_budget,"workstation_policy",lambda:{"storage_root":str(tmp_path),"storage_max_gb":.00001})
    with pytest.raises(storage_budget.StorageLimitError,match="Working-storage limit"):
        tools.pitch_preparation(item,target,semitones=1,rubberband="must-not-run",failure=refusal())
    assert not target.exists()


def test_instrumental_support_restores_voice_in_real_gap_without_cutting_next_word():
    sr=8000;t=np.arange(10*sr)/sr
    bed=np.column_stack([.08*np.sin(2*np.pi*220*t)]*2).astype("float32")
    voice=np.column_stack([.04*np.sin(2*np.pi*700*t)*((t<5)|(t>=6))]*2).astype("float32")
    source=bed+voice
    result,estimate,proof=tools.instrumental_support(source,voice,sr,overlap_frames=5*sr,failure=refusal())
    np.testing.assert_allclose(result[:5*sr],bed[:5*sr],atol=1e-8)
    np.testing.assert_array_equal(result[6*sr:],source[6*sr:])
    np.testing.assert_array_equal(estimate[6*sr:],voice[6*sr:])
    assert proof["native_quiet_restore_measured"] and proof["candidate_only"]
    with pytest.raises(ValueError,match="active native word"):
        tools.instrumental_support(source,voice,sr,overlap_frames=4*sr,failure=refusal())
