"""Bass coverage must be measurable in the source; known losses still fail."""
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter,sosfiltfilt

from superdj.attack_references import retain_mixture_references
from superdj.percussion import verify_blended_fronts


def native_tone_and_fronts(tmp_path,evidence='estimated drums'):
    sr=8000;n=sr*20;t=np.arange(n)/sr;centres=.5+np.arange(38)*.5
    fronts=np.zeros(n);u=np.arange(400)/sr
    pulse=.3*np.sin(2*np.pi*900*(u-.006))*np.exp(-((u-.006)/.0015)**2)
    for stamp in centres:
        i=round(stamp*sr);fronts[i:i+len(pulse)]+=pulse
    drums=np.column_stack([fronts]*2).astype('float32')
    audio=np.column_stack([fronts+.25*np.sin(2*np.pi*80*t)]*2).astype('float32')
    source=tmp_path/'native.wav';sf.write(source,audio,sr,subtype='FLOAT');paths=[]
    for label,band in [('primary',(180,1200)),('bass',(35,180))]:
        filtered=sosfiltfilt(butter(3,band,btype='bandpass',fs=sr,output='sos'),drums[:,0])
        path=tmp_path/(label+'.npz');np.savez(path,templates=np.asarray([filtered[round(c*sr)-96:round(c*sr)+360] for c in centres]));paths.append(str(path))
    s={'evidence':evidence,'attack_templates_path':paths[0],'bass_attack_templates_path':paths[1],
       'attack_band_hz':[180,1200],'events':[{'beat':i+1,'source_seconds':float(c),'output_seconds':float(c)} for i,c in enumerate(centres)],
       'low_frequency_gap_recovery':{'recovered_beats':[]}}
    w={'trim_start_seconds':0.,'source_kick_alignment':{'head':s},
       'input_output_beat_mapping':{'head':[{'source_seconds':0.,'output_seconds':0.},{'source_seconds':20.,'output_seconds':20.}]}}
    retain_mixture_references(source,w,sr)
    gains={'bass_in':np.ones((n,1)),'bass_out':np.zeros((n,1))}
    return sr,audio,w,gains


def test_unmodified_native_tone_needs_no_invented_bass_phase_measurements(tmp_path,monkeypatch):
    from superdj import percussion
    sr,audio,w,gains=native_tone_and_fronts(tmp_path)
    assert not w['source_kick_alignment']['head']['native_mixture_reference']['mixture_bass_visible_beats']
    def unexpected(*args,**kwargs):raise AssertionError('Do not invent source-observability from a new output separation')
    monkeypatch.setattr(percussion,'final_drum_estimates',unexpected)
    same,report=verify_blended_fronts(audio,sr,{},w,gains,tmp_path)
    low=next(c for c in report['checks'] if c['role']=='head:bass_body')
    assert low['native_observable_source_events']==low['coverage_minimum_matches']==0
    assert low['maximum_error_seconds'] is None and low['unverified_source_seconds']
    assert len(report['checks'][0]['recognized'])>=8
    assert np.array_equal(same,audio) and not report['local_combined_waveform_refinements']


def test_mixture_only_baseline_does_not_invent_independent_stem_evidence(tmp_path):
    sr,audio,w,gains=native_tone_and_fronts(tmp_path,evidence='original mixture')
    source=w['source_kick_alignment']['head'];contract=source['native_mixture_reference']
    assert not contract['estimated_drum_discovery_retained'] and not contract['mixture_bass_visible_beats']
    assert not any(k.startswith('estimated_drum_') for k in source)
    _,report=verify_blended_fronts(audio,sr,{},w,gains,tmp_path)
    assert not any(':estimated_' in c['role'] for c in report['checks'])
    assert next(c for c in report['checks'] if c['role']=='head:bass_body')['maximum_error_seconds'] is None


def test_unknown_bass_phase_cannot_hide_a_known_late_primary_front(tmp_path,monkeypatch):
    from superdj import percussion
    sr,audio,w,gains=native_tone_and_fronts(tmp_path)
    late=np.zeros_like(audio);late[64:]=audio[:-64]
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(late,sr,{},w,gains,tmp_path)


def scaffold(tmp_path,contract,recovered=(),independent=False):
    path=tmp_path/'templates.npz';np.savez(path,templates=np.ones((9,456),dtype='float32'))
    events=[{'beat':i+1,'output_seconds':.5+i*.5} for i in range(9)]
    s={'attack_templates_path':str(path),'bass_attack_templates_path':str(path),'attack_band_hz':[180,1200],
       'events':events,'low_frequency_gap_recovery':{'recovered_beats':list(recovered)},'native_mixture_reference':contract}
    if independent:s['estimated_drum_bass_attack_templates_path']=str(path)
    w={'source_kick_alignment':{'head':s},'input_output_beat_mapping':{'head':[{'output_seconds':0.}]}}
    audio=np.zeros((8000*6,2),dtype='float32');gains={'bass_in':np.ones((len(audio),1)),'bass_out':np.zeros((len(audio),1))}
    return audio,w,gains


def test_new_matches_cannot_replace_a_lost_natively_observable_bass_reference(tmp_path,monkeypatch):
    from superdj import percussion,kick_grid
    audio,w,gains=scaffold(tmp_path,{'mixture_bass_visible_beats':[1,4,7]})
    def rows(audio,sr,centres,templates,**kwargs):
        low=kwargs['attack_band_hz']==[35,180]
        return [{'time':t,'eligible':not(low and i==0),'template_correlation':1.} for i,t in enumerate(centres)]
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    # There are eight output bass matches, but one of the three natively
    # observable references is missing. Newly unique waves cannot fill it in.
    with pytest.raises(ValueError,match='fronts remain unverified'):
        verify_blended_fronts(audio,8000,{},w,gains,tmp_path)


@pytest.mark.parametrize('present',[True,False])
def test_certified_recovered_hit_requires_the_natively_visible_independent_view(tmp_path,monkeypatch,present):
    from superdj import percussion,kick_grid
    audio,w,gains=scaffold(tmp_path,{'mixture_bass_visible_beats':[],'bass_visible_beats':[5]},recovered=[5],independent=True)
    def rows(audio,sr,centres,templates,**kwargs):
        low=kwargs['attack_band_hz']==[35,180]
        return [{'time':t,'eligible':not low or (len(templates)==1 and present),'template_correlation':1.} for t in centres]
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    if present:
        _,report=verify_blended_fronts(audio,8000,{},w,gains,tmp_path)
        low=next(c for c in report['checks'] if c['role']=='head:estimated_bass')
        assert low['required_source_seconds']==[2.5] and low['recognized']
    else:
        with pytest.raises(ValueError,match='Recovered bass attacks remain unverified'):
            verify_blended_fronts(audio,8000,{},w,gains,tmp_path)


def test_no_native_baseline_cannot_authorize_a_zero_coverage_bass_view(tmp_path,monkeypatch):
    from superdj import percussion,kick_grid
    audio,w,gains=scaffold(tmp_path,{})
    def rows(audio,sr,centres,templates,**kwargs):
        return [{'time':t,'eligible':kwargs['attack_band_hz']!=[35,180],'template_correlation':1.} for t in centres]
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='fronts remain unverified'):
        verify_blended_fronts(audio,8000,{},w,gains,tmp_path)


@pytest.mark.parametrize('stage',['deck','blend'])
def test_new_bass_correlation_cannot_invent_a_native_attack_or_authorize_correction(tmp_path,monkeypatch,stage):
    from superdj import percussion,kick_grid
    sr,audio,w,gains=native_tone_and_fronts(tmp_path)
    real=kick_grid.template_attack_rows
    def rows(audio,sr,centres,templates,**kwargs):
        if tuple(kwargs.get('attack_band_hz',()))==(35,180):
            return [{'index':i,'time':t+.008,'eligible':True,'template_correlation':.99,'alternative_peak_margin':.1} for i,t in enumerate(centres)]
        return real(audio,sr,centres,templates,**kwargs)
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    if stage=='deck':
        audit=kick_grid.finalize_attack_timing(tmp_path/'native.wav',w,sr)
        assert audit['maximum_measured_drift_seconds']<=.00025
        unknown=[p for p in audit['unverified'] if p.get('native_source_timing_unverified')]
        assert unknown and any(p['output_candidate_delay_seconds']>.007 for p in unknown)
        assert not any(p.get('view')=='bass_body' for p in audit['measured'])
        assert not w.get('fine_attack_timing')
    else:
        monkeypatch.setattr(percussion,'final_drum_estimates',lambda *a,**kw:pytest.fail('No output separation may invent a native clock'))
        unchanged,report=verify_blended_fronts(audio,sr,{},w,gains,tmp_path)
        low=next(c for c in report['checks'] if c['role']=='head:bass_body')
        assert low['maximum_error_seconds'] is None and not low['recognized']
        assert low['native_ambiguous_output_candidates'][0]['candidate_delay_seconds']>.007
        assert np.array_equal(unchanged,audio) and not report['local_combined_waveform_refinements']


def test_known_native_bass_displacement_remains_a_measured_deck_failure(tmp_path,monkeypatch):
    from test_attack_reference_signal import fixture
    from superdj import kick_grid
    sr,path,w=fixture(tmp_path,bed=0.)
    retain_mixture_references(path,w,sr)
    assert 10 in w['source_kick_alignment']['head']['native_mixture_reference']['mixture_bass_visible_beats']
    real=kick_grid.template_attack_rows
    def rows(audio,sr,centres,templates,**kwargs):
        values=real(audio,sr,centres,templates,**kwargs)
        if tuple(kwargs.get('attack_band_hz',()))==(35,180):
            for r in values:
                if r['eligible']:r['time']+=.008
        return values
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    audit=kick_grid.finalize_attack_timing(path,w,sr)
    assert audit['maximum_measured_drift_seconds']>.007
    assert any(p['beat']==10 and p['view']=='bass_body' and p['delay_seconds']>.007 for p in audit['measured'])


def test_fresh_view_cannot_erase_a_strong_natively_supported_late_bass_match(tmp_path,monkeypatch):
    from superdj import percussion,kick_grid
    audio,w,gains=scaffold(tmp_path,{'mixture_bass_visible_beats':list(range(1,10))})
    low_calls=[]
    def rows(audio,sr,centres,templates,**kwargs):
        low=kwargs['attack_band_hz']==[35,180]
        if low:low_calls.append(1)
        first=low and len(low_calls)==1
        return [{'time':t+(.008 if first and i==4 else 0.),'eligible':True,
                 'template_correlation':.91 if first else .999} for i,t in enumerate(centres)]
    monkeypatch.setattr(kick_grid,'template_attack_rows',rows)
    monkeypatch.setattr(percussion,'final_drum_estimates',lambda a,b,*args,**kw:({'drums_out':a,'drums_in':b},[]))
    with pytest.raises(ValueError,match='misses the shared clock'):
        verify_blended_fronts(audio,8000,{},w,gains,tmp_path)
