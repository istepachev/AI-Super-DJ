from copy import deepcopy
import numpy as np
import soundfile as sf
from test_band_hint_front import delayed_body
from superdj.kick_grid import refine_source_controls,rendered_attack_audit
from superdj.attack_references import retain_mixture_references


def test_one_late_click_is_not_hidden_by_an_unchanged_louder_bass_body(tmp_path):
    path,warp,sr,times,nominal=delayed_body(tmp_path)
    report=refine_source_controls(path,warp,sr,evidence={'head':{'path':str(path),'source_start_seconds':0.,'attack_band_hz':[50,400]}})['head']
    retain_mixture_references(path,warp,sr)
    for event in report['events']:event['output_seconds']=event['source_seconds']
    baseline=rendered_attack_audit(path,warp,sr)
    assert baseline['maximum_measured_drift_seconds']<=.001
    audio=sf.read(path,dtype='float32',always_2d=True)[0];broken=audio.copy();bad_beat=next(i for i in range(1,len(times)-1) if i not in set(report['native_mixture_reference']['initial_front_required_beats']));stamp=times[bad_beat];t=np.arange(round(.16*sr))/sr;front=.3*np.sin(2*np.pi*700*t)*np.exp(-t/.004)
    first=round(stamp*sr);broken[first:first+len(front)]-=front[:,None];late=first+round(.008*sr);broken[late:late+len(front)]+=front[:,None]
    target=tmp_path/'one_late_click.wav';sf.write(target,broken,sr,subtype='FLOAT');bad=rendered_attack_audit(target,warp,sr)
    assert bad['maximum_measured_drift_seconds'] is not None and bad['maximum_measured_drift_seconds']>=.006
    # Only the upper click moved; the bass body remains at its original sample.
    assert not np.array_equal(broken,audio)
