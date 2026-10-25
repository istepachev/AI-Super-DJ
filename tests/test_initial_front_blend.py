import numpy as np
import pytest
import soundfile as sf
from test_band_hint_front import delayed_body
from superdj.kick_grid import refine_source_controls
from superdj.attack_references import retain_mixture_references
from superdj.percussion import verify_blended_fronts


def test_transparent_blend_checks_the_front_even_when_the_bass_body_is_unchanged(tmp_path,monkeypatch):
    path,warp,sr,times,_=delayed_body(tmp_path)
    source=refine_source_controls(path,warp,sr,{'head':{'path':str(path),'source_start_seconds':0.,'attack_band_hz':[50,400]}})['head']
    retain_mixture_references(path,warp,sr)
    for event in source['events']:event['output_seconds']=event['source_seconds']
    warp['input_output_beat_mapping']['head'][0]['output_seconds']=0.
    native=sf.read(path,dtype='float32',always_2d=True)[0]
    one=np.ones((len(native),1));zero=one*0
    gains={band+side:one if side=='_in'else zero for band in ('bass','mid','high')for side in ('_in','_out')}
    monkeypatch.setattr('superdj.percussion.final_drum_estimates',lambda a,b,*x,**k:({'drums_out':a,'drums_in':b},[]))
    _,audit=verify_blended_fronts(native,sr,{},warp,gains,tmp_path/'good',components={'head':native,'tail':native*0})
    fronts=next(x for x in audit['checks']if x['role']=='head:initial_front')
    assert fronts['maximum_error_seconds']<=.001
    broken=native.copy();stamp=times[len(times)//2];t=np.arange(round(.16*sr))/sr
    click=.3*np.sin(2*np.pi*700*t)*np.exp(-t/.004);first=round(stamp*sr);late=first+round(.008*sr)
    broken[first:first+len(click)]-=click[:,None];broken[late:late+len(click)]+=click[:,None]
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(broken,sr,{},warp,gains,tmp_path/'bad',components={'head':broken,'tail':broken*0})
