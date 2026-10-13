"""Private reference curves for the user's broad DJ Zhilin handoff."""
from pathlib import Path
import hashlib,json
import numpy as np

def reference_profile(path=None):
    if path is None:
        from .cues import runtime_config
        path=runtime_config().get('zhilin_profile')
    if path:
        source=Path(path).expanduser()
        raw=source.read_bytes();profile=json.loads(raw)
        if profile.get('schema_version')!=1:
            raise ValueError('Unsupported DJ Zhilin profile')
        phase=np.asarray(profile['phase'],dtype=float)
        if len(phase)<3 or phase[0]!=0 or phase[-1]!=1 or np.any(np.diff(phase)<=0):
            raise ValueError('Invalid DJ Zhilin reference phase')
        for key in ('mid_in','high_in','bass_shape'):
            values=np.asarray(profile[key],dtype=float)
            if values.shape!=phase.shape or not np.isfinite(values).all() or np.any(np.diff(values)<-1e-9) or values[0]!=0 or values[-1]!=1:
                raise ValueError('Invalid DJ Zhilin reference curve')
        profile['profile_sha256']=hashlib.sha256(raw).hexdigest()
        return profile
    phase=np.linspace(0,1,257)
    smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
    mid=.45*smooth(phase/.4)+.55*smooth((phase-.4)/.6)
    return {'name':'DJ Zhilin broad fallback','phase':phase.tolist(),'mid_in':mid.tolist(),
        'high_in':smooth(phase).tolist(),'bass_shape':smooth(phase).tolist(),
        'bass_transfer_fraction':[.4875,.6125],'profile_sha256':None,
        'method':'Broad complementary fallback; no private reference configured'}
