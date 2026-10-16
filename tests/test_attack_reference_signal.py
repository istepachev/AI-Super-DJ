"""Discovery stems and rendered mixtures must not be mistaken for one signal."""
from copy import deepcopy
import hashlib
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt

from superdj.attack_references import retain_mixture_references
from superdj.kick_grid import rendered_attack_audit


def fixture(tmp_path,bed):
    sr=8000;t=np.arange(sr*20)/sr;centres=.5+np.arange(38)*.5
    drums=np.zeros(len(t));u=np.arange(900)/sr
    pulse=.2*np.sin(2*np.pi*65*u)*np.exp(-u/.025)+.15*np.random.default_rng(4).normal(size=len(u))*np.exp(-u/.004)
    for stamp in centres:
        i=round(stamp*sr);drums[i:i+len(pulse)]+=pulse
    texture=sosfiltfilt(butter(3,(35,180),btype='bandpass',fs=sr,output='sos'),np.random.default_rng(12).normal(size=len(t)))
    mixture=drums+bed*(.5*np.sin(2*np.pi*109*t+.4*np.sin(t*3))+.6*texture)
    source=tmp_path/'native.wav';sf.write(source,np.column_stack([mixture]*2),sr,subtype='FLOAT')
    paths=[]
    for label,band in [('primary',(180,1200)),('bass',(35,180))]:
        filtered=sosfiltfilt(butter(3,band,btype='bandpass',fs=sr,output='sos'),drums)
        path=tmp_path/f'{label}.npz'
        np.savez(path,templates=np.array([filtered[round(t*sr)-96:round(t*sr)+360] for t in centres]))
        paths.append(str(path))
    alignment={'events':[{'beat':i+1,'source_seconds':float(t),'output_seconds':float(t)} for i,t in enumerate(centres)],
        'evidence':'estimated drums','attack_band_hz':[180,1200],'attack_templates_path':paths[0],
        'bass_attack_templates_path':paths[1],'low_frequency_gap_recovery':{'recovered_beats':[10]}}
    warp={'trim_start_seconds':0.,'source_kick_alignment':{'head':alignment},
        'input_output_beat_mapping':{'head':[{'beat':i,'source_seconds':i*.5,'output_seconds':i*.5} for i in range(41)]}}
    return sr,source,warp


def test_unmodified_native_mix_is_not_repaired_to_match_a_different_stem(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=1.)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    retain_mixture_references(source,warp,sr)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    contract=warp['source_kick_alignment']['head']['native_mixture_reference']
    assert 10 in contract['unresolved_estimated_bass_beats']
    final=rendered_attack_audit(source,warp,sr)
    observed=[p for p in final['measured'] if p['view']=='bass_body']
    assert len(observed)==38 and max(abs(p['delay_seconds']) for p in observed)<1/sr
    assert any(p['beat']==10 and p['view']=='estimated_bass' for p in final['unverified'])


def test_correct_reference_still_detects_a_real_eight_ms_displacement(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=1.)
    retain_mixture_references(source,warp,sr)
    audio,_=sf.read(source,dtype='float32',always_2d=True);delayed=np.zeros_like(audio)
    delayed[64:]=audio[:-64];output=tmp_path/'delayed.wav';sf.write(output,delayed,sr,subtype='FLOAT')
    audit=rendered_attack_audit(output,warp,sr)
    assert audit['maximum_measured_drift_seconds']>.007
    assert any(abs(p['delay_seconds'])>.007 for p in audit['measured'] if p['view']=='bass_body')


def test_visible_native_bass_keeps_an_independent_percussion_check(tmp_path):
    sr,source,warp=fixture(tmp_path,bed=0.)
    retain_mixture_references(source,warp,sr)
    assert 10 in warp['source_kick_alignment']['head']['native_mixture_reference']['bass_visible_beats']
    audio,_=sf.read(source,dtype='float32',always_2d=True)
    center=round(5*sr);end=center+900;original=audio[center:end].copy();audio[center:end]=0
    audio[center+64:end+64]=original
    shifted=tmp_path/'late-bass.wav';sf.write(shifted,audio,sr,subtype='FLOAT')
    audit=rendered_attack_audit(shifted,warp,sr)
    assert any(p['beat']==10 and p['view']=='estimated_bass' and abs(p['delay_seconds'])>.007 for p in audit['measured'])
