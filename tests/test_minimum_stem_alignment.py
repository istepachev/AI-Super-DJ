"""A whole-bar repeat must preserve every prepared stream's sample timeline."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from superdj.compact import arrange_sync
from superdj.minimum import preserve_minimum
from superdj.planner import plan_tracks
from superdj.render import RenderError
from test_compact import record


def inputs(tmp_path):
    sr=8000
    rows=[record('a',125,100),record('b',320,126)]
    rows[0]['downbeats']=[t for t in rows[0]['downbeats'] if t<119]
    plan=arrange_sync(plan_tracks(rows,mode='fixed'),rows)
    t=np.arange(sr*125)/sr
    voice=.03*np.sin(2*np.pi*(390*t+.13*t*t))
    drums=.04*np.sin(2*np.pi*79*t)*(1+.3*np.sin(.83*t))
    values={'path':voice+drums,'vocal_path':voice,'drum_path':drums}
    prepared={'a':{},'b':{'path':'unused'}}
    for key,audio in values.items():
        path=tmp_path/(key+'.wav');sf.write(path,np.column_stack([audio,audio]),sr,subtype='FLOAT')
        prepared['a'][key]=str(path)
    return sr,rows,plan,prepared


def test_short_repeat_preserves_mixture_voice_and_drum_alignment_and_originals(tmp_path):
    sr,rows,plan,prepared=inputs(tmp_path)
    hashes={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in prepared['a'].items()}
    extended,revised,processed=preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    edit=extended['minimum_playback_edits'][0]
    mixture,voice,drums=[sf.read(processed['a'][k],dtype='float32')[0] for k in ('path','vocal_path','drum_path')]
    assert len(mixture)==len(voice)==len(drums)==125*sr+round(edit['added_seconds']*sr)
    np.testing.assert_allclose(mixture,voice+drums,atol=3e-8,rtol=2e-6)
    assert set(edit['aligned_preparation_streams'])=={'path','vocal_path','drum_path'}
    assert extended['sections']['a']['predicted_play_seconds']>=120
    assert revised[0]['duration']>rows[0]['duration']
    for key,path in prepared['a'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==hashes[key]


@pytest.mark.parametrize('broken',['frames','rate'])
def test_misaligned_drum_preparation_stops_before_writing_a_repeat(tmp_path,broken):
    sr,rows,plan,prepared=inputs(tmp_path)
    path=prepared['a']['drum_path'];audio=sf.read(path,dtype='float32')[0]
    sf.write(path,audio[:-1] if broken=='frames' else audio,sr if broken=='frames' else sr*2,subtype='FLOAT')
    with pytest.raises(RenderError,match='same stereo sample timeline'):
        preserve_minimum(plan,rows,prepared,tmp_path/'repeat')
    assert not (tmp_path/'repeat').exists()
