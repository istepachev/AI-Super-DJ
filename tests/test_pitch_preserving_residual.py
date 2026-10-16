"""Accurate attacks must not be bought with an unintended musical pitch bend."""
import hashlib
import inspect
import os
import shutil
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import hilbert
from superdj import kick_grid
from superdj.render import render_plan


def test_old_fractional_delay_can_shift_pitch_despite_passing_rate_gate(tmp_path):
    sr=16000;t=np.arange(sr*4)/sr
    source=tmp_path/'tone.wav';changed=tmp_path/'changed.wav'
    sf.write(source,np.column_stack([.2*np.sin(2*np.pi*440*t)]*2),sr,subtype='FLOAT')
    warp={'fine_attack_timing':{'head':{'times':[0.,1.,2.,3.,4.],
        'offsets':[0.,.01,0.,-.01,0.]}}}
    kick_grid.apply_fine_timing(source,changed,warp,sr)
    audio,_=sf.read(changed,dtype='float64',always_2d=True)
    frequency=np.diff(np.unwrap(np.angle(hilbert(audio[:,0]))))*sr/(2*np.pi)
    measured=float(np.median(frequency[round(1.45*sr):round(1.55*sr)]))
    assert 1200*np.log2(measured/440) < -20
    assert warp['fine_attack_timing']['head']['minimum_rate']>.85


@pytest.mark.parametrize('delay',[.0005,.010])
def test_ordinary_audit_keeps_backend_pcm_even_when_attacks_are_late(tmp_path,monkeypatch,delay):
    source=tmp_path/'rendered.wav';sf.write(source,np.ones((8000,2),dtype='float32')*.1,8000,subtype='FLOAT')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    evidence={'maximum_measured_drift_seconds':delay,'measured':[{'delay_seconds':delay}]}
    monkeypatch.setattr(kick_grid,'rendered_attack_audit',lambda *args:evidence)
    def forbidden(*args,**kwargs):raise AssertionError('Ordinary timing verification bent the pitch')
    monkeypatch.setattr(kick_grid,'fine_align_rendered',forbidden)
    assert kick_grid.finalize_attack_timing(source,{},8000) is evidence
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    assert inspect.signature(render_plan).parameters['allow_pitch_changing_residual'].default is False


def test_advanced_residual_interpolation_requires_explicit_selection(monkeypatch):
    marker={};monkeypatch.setattr(kick_grid,'fine_align_rendered',lambda *args:marker)
    assert kick_grid.finalize_attack_timing('unused',{},8000,allow_pitch_changing_residual=True) is marker
    with pytest.raises(ValueError,match='explicit selection'):
        kick_grid.finalize_attack_timing('unused',{'fine_attack_timing':{'head':{}}},8000)


def test_real_ordinary_render_rejects_late_attacks_instead_of_bending_pitch(tmp_path,monkeypatch):
    from test_sync import RenderedSyncTests
    from superdj import sync
    from superdj.audio import ffmpeg_executable
    from superdj.planner import plan_tracks
    from superdj.transition_quality import TransitionFailure
    rubber=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not rubber:pytest.skip('Actual Rubber Band required')
    records=RenderedSyncTests().fixtures(tmp_path,[120,122])
    actual=kick_grid.rendered_attack_audit
    def late(*args,**kwargs):
        evidence=actual(*args,**kwargs)
        for point in evidence['measured']:
            point['delay_seconds']+=.008;point['observed_seconds']+=.008
        evidence['maximum_measured_drift_seconds']=max(abs(p['delay_seconds']) for p in evidence['measured'])
        return evidence
    def forbidden(*args,**kwargs):raise AssertionError('A failed ordinary render tried pitch-changing interpolation')
    monkeypatch.setattr(kick_grid,'rendered_attack_audit',late)
    monkeypatch.setattr(kick_grid,'fine_align_rendered',forbidden)
    monkeypatch.setattr(sync,'compensate_backend_drift',lambda *args,**kwargs:None)
    output=tmp_path/'ordinary.wav'
    with pytest.raises(TransitionFailure,match='another compatible window'):
        render_plan(plan_tracks(records,mode='fixed'),records,{},output,tmp_path/'work',
            ffmpeg=ffmpeg_executable(),rubberband=rubber,sample_rate=16000,tempo_conflict_policy='sync')
    assert not output.exists()
