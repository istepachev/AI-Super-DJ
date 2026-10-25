import numpy as np
import soundfile as sf
from superdj.front_family import family_coverage
from superdj.source_clock import calibrate_window
from test_band_hint_front import delayed_body
from superdj.kick_grid import refine_source_controls


def test_small_local_front_family_does_not_replace_a_whole_window_clock():
    primary=[{'index':i}for i in range(128)]
    partial=[{'index':i}for i in range(102,120)]
    full=[{'index':i}for i in range(128)if i%5]
    assert not family_coverage(primary,partial)['supported_for_primary_clock']
    assert family_coverage(primary,full)['supported_for_primary_clock']


def test_prepared_mixture_and_drum_views_select_the_same_coherent_leading_family(tmp_path):
    path,first,sr,times,_=delayed_body(tmp_path)
    path,second,sr,times,_=delayed_body(tmp_path)
    with_drums=refine_source_controls(path,first,sr,{'head':{'path':str(path),'source_start_seconds':0.,'attack_band_hz':[50,400]}})['head']
    mixture=refine_source_controls(path,second,sr)['head']
    assert with_drums['attack_band_hz']==mixture['attack_band_hz']==[180,1200]
    assert max(abs(e['source_seconds']-times[e['beat']])for e in mixture['events'])<.006


def test_native_calibration_keeps_primary_when_earlier_clicks_only_cover_the_ending(tmp_path,monkeypatch):
    import superdj.kick_grid as kg
    path=tmp_path/'native.wav';sf.write(path,np.zeros((20*8000,2),dtype='float32'),8000,subtype='FLOAT');times=.5+np.arange(33)*.5
    def events(audio,sr,centres,**kwargs):
        upper=tuple(kwargs.get('attack_band_hz',()))==(180,1200)
        return [{'index':i,'centre':float(t),'time':float(t+(0 if upper else .04)),
            'eligible':not upper or i>=len(centres)-8,'strength':1.,'prominence':100.,'low_power':1.,'window_edge':False}for i,t in enumerate(centres)]
    monkeypatch.setattr(kg,'attack_events',events)
    result=calibrate_window(path,times,band=[50,400])
    assert result['attack_band_hz']==[50,400]
    assert not result['attack_front_comparison']['upper_front_selected']
    assert result['attack_front_comparison']['source_family_coverage']['front_events']==8
