from collections import OrderedDict
from pathlib import Path
from io import BytesIO
import struct,zipfile
import numpy as np
import pytest
import soundfile as sf
from superdj.execution import execution,cached,key_for
from superdj.overlap_identity import dependencies
from superdj.render_index import enable
from superdj.transition_quality import TransitionFailure


def wave(path,*,rate=8000,subtype='FLOAT',delta=0.):
    data=np.column_stack([np.linspace(0.,.2,8000),np.linspace(.3,0.,8000)])
    data[100,0]+=delta;sf.write(path,data,rate,subtype=subtype)
    return data


def add_header(path):
    data=bytearray(path.read_bytes());data+=b'JUNK'+struct.pack('<I',4)+b'date'
    struct.pack_into('<I',data,4,len(data)-8);path.write_bytes(data)


def overlap(a,b,**changes):
    return dict(outgoing={'id':'a','source_path':'source-a','audio_path':str(a)},
                incoming={'id':'b','source_path':'source-b','audio_path':str(b)},
                transition={'from':'a','to':'b','mode':'beatmapped_bass_swap'},sample_rate=8000,**changes)


def identity(session,value):return key_for(dependencies(value,session,OrderedDict()))


def test_rendered_wave_header_dates_do_not_change_signal_identity(tmp_path):
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';wave(a);wave(b)
    with execution()as session:
        before=identity(session,overlap(a,b));raw=key_for(overlap(a,b));add_header(b)
        assert key_for(overlap(a,b))!=raw and identity(session,overlap(a,b))==before


@pytest.mark.parametrize('change',['sample','rate','double'])
def test_actual_samples_rate_and_double_precision_bits_still_change_identity(tmp_path,change):
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';wave(a);wave(b,subtype='DOUBLE'if change=='double'else'FLOAT')
    with execution()as session:
        before=identity(session,overlap(a,b))
        wave(b,rate=16000 if change=='rate'else 8000,subtype='DOUBLE'if change=='double'else'FLOAT',
             delta=1e-12 if change=='double'else.0001)
        assert identity(session,overlap(a,b))!=before


def npz(path,date,array):
    stream=BytesIO();np.save(stream,array,allow_pickle=False)
    with zipfile.ZipFile(path,'w')as archive:
        entry=zipfile.ZipInfo('templates.npy',date_time=date);archive.writestr(entry,stream.getvalue())


def test_npz_dates_are_ignored_but_array_values_dtype_layout_and_names_are_not(tmp_path):
    p=tmp_path/'refs.npz';a=np.arange(24,dtype='float32').reshape(6,4)
    npz(p,(2020,1,1,0,0,0),a)
    with execution()as session:
        before=identity(session,{'templates':str(p)})
        npz(p,(2021,1,1,0,0,0),a);assert identity(session,{'templates':str(p)})==before
        for changed in(a+.00001,a.astype('float64'),np.asfortranarray(a)):
            npz(p,(2021,1,1,0,0,0),changed);assert identity(session,{'templates':str(p)})!=before


def test_structured_dtype_names_are_preserved(tmp_path):
    p=tmp_path/'refs.npz';a=np.zeros(4,dtype=[('time','f8'),('gain','f8')]);np.savez(p,values=a)
    with execution()as session:
        before=identity(session,{'templates':str(p)})
        np.savez(p,values=np.zeros(4,dtype=[('gain','f8'),('time','f8')]))
        assert identity(session,{'templates':str(p)})!=before


def test_original_source_wave_keeps_ordinary_full_file_identity(tmp_path):
    p=tmp_path/'native.wav';wave(p)
    with execution()as session:
        before=identity(session,{'source_path':str(p)});add_header(p)
        assert identity(session,{'source_path':str(p)})!=before


def test_header_only_rejection_is_replayed_across_restart_without_new_dsp(tmp_path):
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';wave(a);wave(b);calls=[]
    for turn in range(2):
        if turn:add_header(b)
        with execution(tmp_path/'cache')as session:
            index=enable(session,'same programme')
            def fail():calls.append(1);raise TransitionFailure('a','b','bad actual mixture',{'delay':.004})
            with pytest.raises(TransitionFailure)as error:cached('overlap',overlap(a,b),fail,session.work)
            if turn:assert error.value.evidence['render_index_replayed_rejection']and index.metrics['fresh_candidates']==0
    assert len(calls)==1


def test_positive_signal_alias_still_checks_every_cached_artifact(tmp_path):
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';wave(a);wave(b);calls=[]
    with execution(tmp_path/'cache')as session:
        enable(session,'same programme')
        def produce():
            calls.append(1);out=session.work/'mixed.wav';sf.write(out,np.zeros((8000,2)),8000,subtype='FLOAT');return {'path':str(out)}
        first=cached('overlap',overlap(a,b),produce,session.work);add_header(b)
        again=cached('overlap',overlap(a,b),produce,session.work)
        assert len(calls)==1 and first['path']==again['path']
        Path(first['path']).write_bytes(b'corrupt cached artifact')
        repaired=cached('overlap',overlap(a,b),produce,session.work)
        assert len(calls)==2 and sf.info(repaired['path']).frames==8000


def test_changed_clock_and_processing_controls_do_not_alias(tmp_path):
    a=tmp_path/'a.wav';b=tmp_path/'b.wav';wave(a);wave(b)
    with execution()as session:
        before=identity(session,overlap(a,b,gain=.5,clock=[0.,1.]))
        assert identity(session,overlap(a,b,gain=.6,clock=[0.,1.]))!=before
        assert identity(session,overlap(a,b,gain=.5,clock=[0.,1.0001]))!=before
