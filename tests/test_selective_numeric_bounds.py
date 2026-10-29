import copy
import numpy as np
import pytest
from superdj.selective_mid import decide,apply_report
from superdj.mixing import envelopes

def report():
    sr=8000;t=np.arange(round(sr*61.987))/sr
    u=np.linspace(0,1,len(t));phase=u+.017*np.sin(2*np.pi*u)
    a=np.repeat((.06*np.sin(2*np.pi*430*t))[:,None],2,axis=1)
    b=np.repeat((.06*np.sin(2*np.pi*2100*t))[:,None],2,axis=1)
    value=decide(a,b,sr,beat_phase=phase,beats=128,enabled=True)
    gains,_=envelopes(len(a),128,style='zhilin',beat_phase=phase)
    return value,gains,phase

def test_generated_weighted_support_remains_usable_on_nonuniform_clock():
    value,gains,phase=report();strengths=np.asarray(value['strengths'])
    assert value['applied'] and np.isfinite(strengths).all()
    assert strengths.min()>=0 and strengths.max()<=1
    position=gains['mid_in']/(gains['mid_in']+gains['mid_out'])
    result=apply_report(gains,{'selective_mid_energy':value},position=position,beat_phase=phase)
    for side in ('out','in'):
        for cell in range(8):
            g=result[f'mid_cell_{cell}_{side}']
            assert np.isfinite(g).all() and g.min()>=0 and g.max()<=1
            assert np.all(np.diff(g[:,0])<=1e-14) if side=='out' else np.all(np.diff(g[:,0])>=-1e-14)

def test_reader_still_refuses_outward_ulp_in_supplied_evidence():
    value,gains,phase=report();bad=copy.deepcopy(value)
    bad['strengths'][0][0][1]=float(np.nextafter(1.,2.))
    position=gains['mid_in']/(gains['mid_in']+gains['mid_out'])
    with pytest.raises(ValueError,match='actual overlap'):
        apply_report(gains,{'selective_mid_energy':bad},position=position,beat_phase=phase)
