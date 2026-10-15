"""Portable reference curves for DJ Zhilin's broad frequency handoff."""
from pathlib import Path
from importlib.resources import files
import hashlib,json
import numpy as np

def reference_profile(path=None, *, runtime=None):
    source_kind='explicit'
    if path is None:
        if runtime is None:
            from .cues import runtime_config
            runtime=runtime_config()
        path=runtime.get('zhilin_profile')
        source_kind='runtime'
    if path:
        source=Path(path).expanduser()
        try:
            raw=source.read_bytes()
        except OSError as exc:
            raise ValueError('Selected DJ Zhilin reference profile is unavailable; choose an existing zhilin_profile JSON or remove that setting to use the packaged approved reference') from exc
    else:
        source_kind='packaged'
        try:
            raw=files('superdj').joinpath('profiles','zhilin-reference-v1.json').read_bytes()
        except OSError as exc:
            raise ValueError('Packaged DJ Zhilin reference profile is unavailable; reinstall SuperDJ or configure an existing zhilin_profile JSON') from exc
    profile=json.loads(raw)
    if profile.get('schema_version')!=1:
        raise ValueError('Unsupported DJ Zhilin profile')
    phase=np.asarray(profile['phase'],dtype=float)
    if len(phase)<3 or not np.isfinite(phase).all() or phase[0]!=0 or phase[-1]!=1 or np.any(np.diff(phase)<=0):
        raise ValueError('Invalid DJ Zhilin reference phase')
    for key in ('mid_in','high_in','bass_shape'):
        values=np.asarray(profile[key],dtype=float)
        if values.shape!=phase.shape or not np.isfinite(values).all() or np.any(np.diff(values)<-1e-9) or values[0]!=0 or values[-1]!=1:
            raise ValueError('Invalid DJ Zhilin reference curve')
    profile['profile_sha256']=hashlib.sha256(raw).hexdigest()
    profile['profile_source']=source_kind
    return profile
