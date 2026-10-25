import numpy as np
import pytest
from scipy.signal import butter,sosfiltfilt
from superdj.percussion import verify_blended_fronts


def test_known_upper_attack_cannot_hide_behind_zero_bass_gain(tmp_path,monkeypatch):
    sr=8000;n=sr*20;centres=.5+np.arange(36)*.5;t=np.arange(456)/sr
    pulse=.15*np.random.default_rng(929).normal(size=len(t))*np.exp(-t/.004);native=np.zeros((n,2),dtype='float32')
    for stamp in centres:
        j=round(stamp*sr);native[j:j+len(pulse)]+=pulse[:,None]
    filtered=sosfiltfilt(butter(3,(180,1200),btype='bandpass',fs=sr,output='sos'),native[:,0]);templates=np.array([filtered[round(x*sr)-96:round(x*sr)+360]for x in centres]);path=tmp_path/'native.npz';np.savez(path,templates=templates)
    source={'attack_templates_path':str(path),'attack_band_hz':[180,1200],'events':[{'beat':i+1,'output_seconds':float(x)}for i,x in enumerate(centres)]};warp={'source_kick_alignment':{'tail':source},'input_output_beat_mapping':{'tail':[{'output_seconds':0.}]}}
    bad=native.copy();first=round(centres[-5]*sr);bad[first:first+len(pulse)]=0;bad[first+64:first+64+len(pulse)]=native[first:first+len(pulse)]
    oldbass=np.ones((n,1));oldbass[n//2:]=0;g={'bass_out':oldbass,'bass_in':1-oldbass,'mid_out':np.ones((n,1))*.5,'mid_in':np.ones((n,1))*.5,'high_out':np.ones((n,1))*.5,'high_in':np.ones((n,1))*.5}
    own=bad*.5;zero=np.zeros_like(own);monkeypatch.setattr('superdj.percussion.final_drum_estimates',lambda a,b,*x,**k:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(own,sr,warp,{},g,tmp_path,components={'tail':own,'head':zero})
