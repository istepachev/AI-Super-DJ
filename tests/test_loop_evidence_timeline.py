from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pytest
import soundfile as sf
from superdj.loop_evidence import render_records


def fixture(tmp_path):
    sr=8000;item={}
    for key,gain in [('path',1.),('vocal_path',.2),('drum_path',.5)]:
        path=tmp_path/(key+'.wav');sf.write(path,np.ones((20*sr,2),dtype='float32')*.01*gain,sr,subtype='FLOAT');item[key]=str(path)
    edit={'method':'verified phrase repeat with complementary-band join','loop_start_seconds':4.,'insertion_source_seconds':8.,'loop_seconds':4.,'repeat_count':2,'added_seconds':8.}
    rows=[{'id':'a','timeline_edit':edit,'kick_evidence':{'tail':{'path':'native drums','source_start_seconds':0.,'attack_band_hz':[50,400]}},'vocal_stem':{'profile':[{'time':5.,'foreground_risk':.2},{'time':10.,'foreground_risk':.7}]}}]
    return rows,{'a':item}


def test_edited_evidence_and_voice_profile_use_one_actual_pcm_timeline(tmp_path):
    rows,prepared=fixture(tmp_path);before=deepcopy(rows);hashes={p:hashlib.sha256(open(p,'rb').read()).hexdigest()for p in prepared['a'].values()}
    actual,ledger=render_records(rows,prepared)
    assert rows==before
    assert actual[0]['kick_evidence']['tail']['path']==prepared['a']['drum_path']
    assert actual[0]['kick_evidence']['tail']['attack_band_hz']==[50,400]
    assert [p['time']for p in actual[0]['vocal_stem']['profile']]==[5.,9.,13.,18.]
    assert ledger[0]['native_evidence']['path']=='native drums'
    assert all(hashlib.sha256(open(p,'rb').read()).hexdigest()==v for p,v in hashes.items())


def test_second_render_does_not_shift_the_profile_again(tmp_path):
    rows,prepared=fixture(tmp_path);first,_=render_records(rows,prepared);second,ledger=render_records(first,prepared)
    assert second==first and not ledger


def test_plain_sources_keep_existing_evidence_and_object_identity():
    rows=[{'id':'a','kick_evidence':{'tail':{'path':'native'}}}]
    actual,ledger=render_records(rows,{})
    assert actual is rows and not ledger

@pytest.mark.parametrize('missing',[False,True])
def test_missing_or_mismatched_edited_drums_fail_before_sync(tmp_path,missing):
    rows,prepared=fixture(tmp_path)
    if missing:prepared['a'].pop('drum_path')
    else:sf.write(prepared['a']['drum_path'],np.zeros((12345,2),dtype='float32'),8000,subtype='FLOAT')
    before=deepcopy(rows)
    with pytest.raises(ValueError):render_records(rows,prepared)
    assert rows==before


def test_reused_edit_rebinds_fresh_matching_streams_without_repeating_profile_shift(tmp_path):
    rows,prepared=fixture(tmp_path);first,_=render_records(rows,prepared)
    folder=tmp_path/'fresh';folder.mkdir();import shutil
    fresh={'a':{}}
    for key,path in prepared['a'].items():
        target=folder/Path(path).name;shutil.copy2(path,target);fresh['a'][key]=str(target)
    second,ledger=render_records(first,fresh)
    assert second[0]['kick_evidence']['tail']['path']==fresh['a']['drum_path']
    assert second[0]['vocal_stem']['profile']==first[0]['vocal_stem']['profile'] and ledger
