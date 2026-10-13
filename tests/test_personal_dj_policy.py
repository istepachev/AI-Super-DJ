import json
from pathlib import Path
import numpy as np
import pytest
from superdj.history import register_manifest,unseen_files
from superdj.storage import select_tracks
from superdj.naming import dated_name
from superdj.mixing import envelopes,blend
from superdj.history import import_cue_folder,read_history
from superdj.delivery import make_cue,publish_pair

def test_cue_history_and_delivery_preserve_old_audio_privately(tmp_path):
    ready=tmp_path/'ready';ready.mkdir()
    audio=tmp_path/'new.flac';audio.write_bytes(b'new verified audio')
    manifest={'exports':{'flac':{'path':str(audio)}},
              'selection_and_analysis':[{'id':'one','path':'Artist - Song.mp3'}],
              'render':{'tracks':[{'id':'one','start_seconds':80.5}]}}
    cue=make_cue(manifest)
    (ready/audio.name).write_bytes(b'previous audio')
    (ready/audio.with_suffix('.cue').name).write_text('previous cue')
    backup=tmp_path/'backup'
    result=publish_pair(audio,cue,ready,replace=True,backup=backup)
    assert (backup/audio.name).read_bytes()==b'previous audio'
    assert Path(result['flac']).read_bytes()==audio.read_bytes()
    assert 'INDEX 01 01:20:38' in Path(result['cue']).read_text(encoding='utf-8-sig')
    history=tmp_path/'history.json'
    assert import_cue_folder(ready,history)['tracks']==1
    assert len(read_history(history)['sources'])==1
    assert {p.name for p in ready.iterdir()}=={'new.flac','new.cue'}

def test_history_excludes_renamed_copy_and_other_encoding_by_name(tmp_path):
    sources=[]
    for index,title in enumerate(('Artist - Song (Remix)','Other - B','Third - C')):
        p=tmp_path/(title+'.mp3');p.write_bytes(bytes([index+1])*23);sources.append(p)
    output=tmp_path/'completed.flac';output.write_bytes(b'fixture')
    history=tmp_path/'history.json'
    manifest={'selection_and_analysis':[{'path':str(p)} for p in sources],
        'exports':{'flac':{'path':str(output),'full_decode_ok':True}}}
    register_manifest(manifest,history)
    copied=tmp_path/'Renamed.flac';copied.write_bytes(sources[0].read_bytes())
    other_encoding=tmp_path/'Artist - Song (Remix).flac';other_encoding.write_bytes(b'different encoding')
    alternate_remix=tmp_path/'Artist - Song (Different Remix).mp3';alternate_remix.write_bytes(b'new remix')
    files,excluded=unseen_files([copied,other_encoding,alternate_remix],history)
    assert files==[alternate_remix] and len(excluded)==2
    folder=tmp_path/'selection';folder.mkdir()
    (folder/copied.name).write_bytes(copied.read_bytes())
    with pytest.raises(ValueError,match='previously used'):
        select_tracks(folder,1,4,history=history)
    assert len(select_tracks(folder,1,4,history=history,allow_repeats=True)['tracks'])==1

def test_date_first_name_keeps_explicit_date_and_cyrillic():
    assert dated_name('Вечериночка_Мажор','2026_10_12')=='2026_10_12_Вечериночка_Мажор'
    assert dated_name('2026_10_11_ДР_Нади','2026_10_12')=='2026_10_11_ДР_Нади'

def test_zhilin_broad_mids_retain_music_before_and_after_bass():
    p=np.linspace(0,1,8193)
    gains,report=envelopes(len(p),128,beat_phase=p,style='zhilin')
    assert gains['mid_in'][int(.25*len(p)),0]>.2
    assert gains['mid_out'][int(.70*len(p)),0]>.1
    assert report['bass_handoff_beats']==pytest.approx(16)
    for band in ('bass','mid','high'):
        np.testing.assert_allclose(gains[band+'_in']+gains[band+'_out'],1,atol=1e-12)
    assert gains['mid_in'][0,0]==0 and gains['mid_out'][-1,0]==0
    sr=16000;t=np.arange(sr*3)/sr
    signal=.05*(np.sin(2*np.pi*70*t)+np.sin(2*np.pi*1000*t)+np.sin(2*np.pi*5000*t))
    audio=np.column_stack([signal,signal]).astype(np.float32)
    mixed,_=blend(audio,audio,sr,beats=128,style='zhilin')
    np.testing.assert_allclose(mixed,audio,atol=3e-8)

def test_reviewed_mid_change_preserves_bass_and_highs_and_separate_bass_change_preserves_mids():
    phase=np.linspace(0,1,8193)
    original,_=envelopes(len(phase),128,beat_phase=phase,style='zhilin')
    short,report=envelopes(len(phase),128,beat_phase=phase,style='zhilin',mid_handoff_bars=8)
    assert report['mid_handoff_fraction']==pytest.approx([.425,.675])
    for key in ('bass_in','bass_out','high_in','high_out'):
        np.testing.assert_array_equal(short[key],original[key])
    shorter_bass,report=envelopes(len(phase),128,beat_phase=phase,style='zhilin',bass_handoff_beats=4)
    assert report['bass_handoff_beats']==pytest.approx(4)
    for key in ('mid_in','mid_out','high_in','high_out'):
        np.testing.assert_array_equal(shorter_bass[key],original[key])
    for band in ('bass','mid','high'):
        np.testing.assert_allclose(shorter_bass[band+'_in']+shorter_bass[band+'_out'],1)
    with pytest.raises(ValueError):envelopes(len(phase),128,style='zhilin',mid_handoff_bars=33)
