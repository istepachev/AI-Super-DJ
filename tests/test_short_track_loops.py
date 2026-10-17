"""Short native play survives; only supported, reversible loops extend it."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pytest
import soundfile as sf

from superdj.compact import arrange_sync
from superdj.loops import assess_join,join_tail,repeat_file,choose_repeat
from superdj.minimum import preserve_minimum
from superdj.minimum_timeline import native_prepared,native_time,native_records,native_plan
from superdj.planner import plan_tracks
from superdj.render import RenderError,_validate_playback_budget
from test_compact import record
from test_minimum_stem_alignment import inputs


@pytest.mark.parametrize('duration',[112,92,64])
def test_110_90_and_60_second_native_play_does_not_require_a_loop(tmp_path,duration):
    row=record('short',duration,120);row['protected_ranges']=[[10.,20.]]
    path=tmp_path/'short.wav';t=np.arange(duration*8000)/8000
    sf.write(path,np.column_stack([.04*np.sin(2*np.pi*300*t)]*2),8000,subtype='FLOAT')
    plan=arrange_sync(plan_tracks([row],mode='fixed'),[row]);prepared={'short':{'path':str(path)}}
    assert 60<=plan['sections']['short']['predicted_play_seconds']<120
    before=deepcopy((plan,row,prepared));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    result,rows,files=preserve_minimum(plan,[row],prepared,tmp_path/'optional')
    assert not result['minimum_playback_edits'] and result['short_track_loop_reviews'][0]['short_play_retained']
    assert result['sections']['short']['predicted_play_seconds']==plan['sections']['short']['predicted_play_seconds']
    assert files==prepared and rows==[row] and not (tmp_path/'optional').exists()
    assert (plan,row,prepared)==before and hashlib.sha256(path.read_bytes()).hexdigest()==digest


def test_failed_optional_pcm_join_keeps_the_native_track(tmp_path,monkeypatch):
    _,rows,plan,prepared=inputs(tmp_path)
    monkeypatch.setattr('superdj.short_tracks.choose_repeat',lambda *a,**k:(None,'incompatible native melody'))
    result,native,files=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    assert files==prepared and native==rows and not result['minimum_playback_edits']
    assert result['sections']['a']['minimum_play_seconds']==60


def test_a_source_below_one_minute_needs_a_supported_extension(tmp_path,monkeypatch):
    _,rows,plan,prepared=inputs(tmp_path);plan['sections']['a']['predicted_play_seconds']=45.
    monkeypatch.setattr('superdj.short_tracks.choose_repeat',lambda *a,**k:(None,'incompatible native melody'))
    with pytest.raises(RenderError,match='one-minute floor'):
        preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    assert not (tmp_path/'repeat').exists()


def test_harmonic_or_bass_conflict_cannot_be_hidden_by_a_short_fade():
    sr=8000;t=np.arange(sr)/sr
    a=np.column_stack([.08*np.sin(2*np.pi*80*t)+.04*np.sin(2*np.pi*400*t)]*2).astype('float32')
    harmony=np.column_stack([.08*np.sin(2*np.pi*80*t)+.04*np.sin(2*np.pi*500*t)]*2).astype('float32')
    assert assess_join(a,harmony,sr,.5)[0] is None
    assert assess_join(a,-a,sr,.5)[0] is None
    quality,reason=assess_join(a,a,sr,.5)
    assert quality['passed'] and quality['maximum_energy_hole_seconds']==0 and reason is None
    np.testing.assert_allclose(join_tail(a,a,sr,.5),a,atol=1e-8)


def test_every_repeated_join_has_native_endpoints_and_an_exact_pcm_inverse(tmp_path):
    sr=8000;t=np.arange(sr*14)/sr
    pcm=np.column_stack([.07*np.sin(2*np.pi*80*t)+.03*np.sin(2*np.pi*400*t)]*2).astype('float32')
    source=tmp_path/'native.wav';sf.write(source,pcm,sr,subtype='FLOAT')
    edit={'method':'verified phrase repeat with complementary-band join','loop_start_seconds':4.,
        'insertion_source_seconds':8.,'loop_seconds':4.,'repeat_count':3,'added_seconds':12.,
        'bars_repeated':2,'join_frames':sr,'beat_seconds':.5,'sample_rate':sr,
        'original_source_duration_seconds':14.,'aligned_preparation_streams':['path']}
    output=tmp_path/'loop.wav'
    context=repeat_file(source,output,edit,context_path=tmp_path/'native_join.npy')
    edit['native_join_contexts']={'path':context}
    result=sf.read(output,dtype='float32')[0]
    assert len(result)==len(pcm)+12*sr
    for stamp in (8,12,16):
        # Last preroll sample and first copied sample are natural neighbours
        # from the same native phrase, with no temporal splice gap.
        np.testing.assert_array_equal(result[stamp*sr-1],pcm[4*sr-1])
        np.testing.assert_array_equal(result[stamp*sr],pcm[4*sr])
    np.testing.assert_array_equal(result[20*sr:],pcm[8*sr:])
    for stamp in (8.2,12.2,16.2):assert native_time(stamp,edit)==pytest.approx(4.2)
    assert native_time(20.2,edit)==pytest.approx(8.2)
    prepared={'a':{'path':str(output),'provenance':{'timeline_edit':edit}}}
    restored=native_prepared(prepared,tmp_path/'restored')
    np.testing.assert_array_equal(sf.read(restored['a']['path'],dtype='float32')[0],pcm)


def test_source_timeline_and_neural_probabilities_round_trip_with_multiple_cycles(tmp_path):
    _,rows,plan,prepared=inputs(tmp_path)
    # A lower native duration objective requests several cycles from a good
    # phrase; no fake beats are introduced into the returned native record.
    plan['sections']['a']['predicted_play_seconds']=65.
    row=rows[0];row['grid']['neural_beat_probability_by_beat']=[.99]*len(row['beats'])
    row['grid']['neural_downbeat_probability_by_beat']=[.99 if n%4==0 else .01 for n in range(len(row['beats']))]
    result,edited,files=preserve_minimum(plan,rows,prepared,tmp_path/'loop')
    assert result['minimum_playback_edits']
    recovered=native_records(edited);restored_plan=native_plan(result)
    np.testing.assert_allclose(recovered[0]['beats'],row['beats'],atol=1e-12,rtol=0)
    assert recovered[0]['grid']==row['grid']
    assert restored_plan['sections']['a']['predicted_play_seconds']==pytest.approx(65.)
    assert files['a']['path']!=prepared['a']['path']


def test_optional_loop_uses_supported_local_neural_windows(tmp_path):
    from test_local_meter_windows import locally_supported
    row=locally_supported();row['duration']=row['duration_seconds']=125.
    row['beats']=[v for v in row['beats'] if v<125]
    row['downbeats']=[v for v in row['downbeats'] if v<125]
    for name in ('neural_beat_probability_by_beat','neural_downbeat_probability_by_beat'):
        row['grid'][name]=row['grid'][name][:len(row['beats'])]
    sr=8000;t=np.arange(sr*125)/sr;path=tmp_path/'local.wav'
    sf.write(path,np.column_stack([.05*np.sin(2*np.pi*80*t)+.03*np.sin(2*np.pi*400*t)]*2),sr,subtype='FLOAT')
    plan={'order':['local'],'edges':[]};section={'start':.25,'end':120.25,'predicted_play_seconds':100.}
    choice,reason=choose_repeat(plan,row,0,section,path)
    assert choice and choice['join_quality']['passed'] and reason is None


def test_automatic_folder_selection_reports_45_second_sources_instead_of_aborting_valid_tracks(tmp_path):
    from superdj.make import _folder_short_sources
    paths=[tmp_path/name for name in ('short.wav','one_minute.wav','long.wav')]
    probes=[{'path':str(p),'duration_seconds':duration} for p,duration in zip(paths,(45,60,180))]
    selected,excluded=_folder_short_sources(paths,probes)
    assert selected==paths[1:] and excluded[0]['path']==str(paths[0]) and excluded[0]['duration_seconds']==45
    assert len(excluded)==1


def test_actual_full_playback_respects_the_one_minute_floor():
    with pytest.raises(RenderError,match='falls below 60s'):
        _validate_playback_budget({'id':'short','output_frames':59*8000},{'minimum_play_seconds':60},8000)
    for seconds in (60,90,110,300):
        _validate_playback_budget({'id':'short','output_frames':seconds*8000},{'minimum_play_seconds':60},8000)
