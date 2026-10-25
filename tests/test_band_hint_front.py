from pathlib import Path
from copy import deepcopy
import numpy as np
import soundfile as sf
import pytest
from superdj.kick_grid import refine_source_controls
from superdj.source_clock import calibrate_window

def delayed_body(tmp_path):
    sr=16000;times=np.arange(.3,17.4,.503);nominal=np.round(times/.02)*.02+.014;n=18*sr
    audio=np.zeros((n,2),dtype='float32');t=np.arange(round(.16*sr))/sr
    front=.3*np.sin(2*np.pi*700*t)*np.exp(-t/.004);body=1.2*np.sin(2*np.pi*110*t)*np.exp(-t/.032)
    for stamp in times:
        first=round(stamp*sr);audio[first:first+len(front)]+=front[:,None]
        first=round((stamp+.04)*sr);audio[first:first+len(body)]+=body[:,None]
    path=tmp_path/'source.wav';sf.write(path,audio,sr,subtype='FLOAT')
    phases=np.arange((len(times)-1)*4+1)/4;sources=np.interp(phases,np.arange(len(times)),nominal);outputs=.3+phases*.5
    anchors=[{'source_frame':0,'output_frame':0,'source_seconds':0.,'output_seconds':0.,'role':'clip_start'}]
    anchors.extend({'source_frame':round(a*sr),'output_frame':round(b*sr),'source_seconds':a,'output_seconds':b,'role':'head_sync_control'}for a,b in zip(sources,outputs))
    anchors.append({'source_frame':n,'output_frame':n,'source_seconds':18.,'output_seconds':18.,'role':'clip_end'})
    warp={'anchors':anchors,'trim_start_seconds':0.,'output_frames':n,'input_output_beat_mapping':{'head':[{'beat':i,'source_seconds':float(a),'output_seconds':float(.3+i*.5)}for i,a in enumerate(nominal)],'tail':[]}}
    return path,warp,sr,times,nominal

@pytest.mark.parametrize('band',[[50,400],[100,1400]])
def test_band_hint_does_not_skip_coherent_leading_attack(tmp_path,band):
    path,warp,sr,times,nominal=delayed_body(tmp_path);before=path.read_bytes()
    report=refine_source_controls(path,warp,sr,evidence={'head':{'path':str(path),'source_start_seconds':0.,'attack_band_hz':band}})['head']
    assert report.get('attack_front_comparison',{}).get('upper_front_selected')
    assert report['attack_band_hz']==[180,1200]
    assert max(abs(e['source_seconds']-times[e['beat']])for e in report['events'])<.006
    assert path.read_bytes()==before

@pytest.mark.parametrize('band',[[50,400],[100,1400]])
def test_source_cut_calibration_uses_same_front_with_band_hint(tmp_path,band):
    path,warp,sr,times,nominal=delayed_body(tmp_path)
    result=calibrate_window(path,nominal,band=band)
    assert result is not None
    assert result['attack_band_hz']==[180,1200]
    assert np.max(np.abs(np.array(result['times'])-times))<.006
