"""An uncertain source window must not silently become a PCM patch montage."""
import os
import shutil
import numpy as np
import pytest
import soundfile as sf

from superdj.planner import plan_tracks
from superdj.render import render_plan
from superdj.transition_quality import TransitionFailure


def test_real_render_rejects_a_window_instead_of_copying_native_fragments(tmp_path,monkeypatch):
    from superdj import kick_grid
    ffmpeg=os.getenv('SUPERDJ_FFMPEG') or shutil.which('ffmpeg')
    rubberband=os.getenv('SUPERDJ_RUBBERBAND') or shutil.which('rubberband')
    if not ffmpeg or not rubberband:pytest.skip('Real FFmpeg/Rubber Band needed')
    sr=8000;records=[]
    for identity,bpm in [('a',120),('b',122)]:
        beats=np.arange(.25,89.9,60/bpm);audio=np.zeros((90*sr,2),dtype='float32')
        pulse=.2*np.exp(-np.arange(320)/30)*np.sin(2*np.pi*80*np.arange(320)/sr)
        for stamp in beats:
            i=round(stamp*sr)
            audio[i:i+len(pulse)]+=pulse[:,None]
        path=tmp_path/f'{identity}.wav';sf.write(path,audio,sr,subtype='FLOAT')
        records.append(dict(id=identity,path=str(path),duration=90.,bpm=bpm,bpm_confidence=1.,
            key_camelot='8A',key_confidence=1.,energy=.5,beats=beats.tolist(),downbeats=beats[::4].tolist(),
            grid_status='reviewed',grid_confidence=1.,grid={'meter':4,'human_reviewed':True}))
    refine=kick_grid.refine_source_controls
    def unsupported(source,warp,rate,evidence=None):
        refine(source,warp,rate,evidence)
        warp['source_attack_controls_unapplied']=True
    def no_copy(*args,**kwargs):
        raise AssertionError('Ordinary rendering tried to transplant source PCM')
    monkeypatch.setattr(kick_grid,'refine_source_controls',unsupported)
    monkeypatch.setattr(kick_grid,'preserve_rendered_attacks',no_copy)
    output=tmp_path/'mix.wav'
    with pytest.raises(TransitionFailure,match='another compatible window'):
        render_plan(plan_tracks(records,mode='fixed'),records,{},output,tmp_path/'work',
            ffmpeg=ffmpeg,rubberband=rubberband,sample_rate=sr,tempo_conflict_policy='sync',transition_bars=8)
    assert not output.exists()
