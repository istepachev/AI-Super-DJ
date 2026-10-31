"""Verified whole-clip integer delay; no local timing or pitch interpolation."""
from pathlib import Path
from uuid import uuid4
import math
import numpy as np
import soundfile as sf

from .integrity import _timing_limit_exceeded
from .source_observability import required_bass_views


def shift_stream(source, destination, advance_frames):
    """Translate every channel identically, padding only the bounded edges."""
    if isinstance(advance_frames,bool)or not isinstance(advance_frames,int):raise ValueError('Integer sample offset required')
    with sf.SoundFile(source)as reader:
        if reader.subtype not in ('FLOAT','DOUBLE')or abs(advance_frames)>=reader.frames:
            raise ValueError('Phase translation needs a complete floating-point stream')
        dtype='float64'if reader.subtype=='DOUBLE'else'float32'
        with sf.SoundFile(destination,'w',samplerate=reader.samplerate,channels=reader.channels,subtype=reader.subtype)as writer:
            if advance_frames<0:writer.write(np.zeros((-advance_frames,reader.channels),dtype=dtype))
            reader.seek(max(0,advance_frames));left=reader.frames-abs(advance_frames)
            while left:
                block=reader.read(min(left,262144),dtype=dtype,always_2d=True)
                if not len(block)or not np.isfinite(block).all():raise ValueError('Invalid phase-translation PCM')
                writer.write(block);left-=len(block)
            if advance_frames>0:writer.write(np.zeros((advance_frames,reader.channels),dtype=dtype))


def apply_aligned_stream(path, phase):
    if not phase or not phase.get('applied'):return
    path=Path(path);candidate=path.with_name(path.stem+'-phase-'+uuid4().hex+'.wav')
    try:
        info=sf.info(path)
        if info.samplerate!=phase['sample_rate']:raise ValueError('Aligned phase stream has a different sample rate')
        shift_stream(path,candidate,phase['advance_frames'])
        candidate.replace(path)
    finally:candidate.unlink(missing_ok=True)


def _coverage(audit,warp):
    points=audit.get('measured',[])
    if set(warp.get('source_kick_alignment',{}))!={'head','tail'}:return False
    for role,row in warp['source_kick_alignment'].items():
        seen=lambda view:{p['beat']for p in points if p['role']==role and p.get('view')==view}
        if len(row.get('events',[]))<8 or len(seen('primary'))<8:return False
        required=required_bass_views(row)
        if not required['mixture']<=seen('bass_body')or not required['estimate']<=seen('estimated_bass'):return False
        if not set(row.get('native_mixture_reference',{}).get('initial_front_required_beats',[]))<=seen('initial_front'):return False
    return True


def align_constant_phase(path,warp,audit,sample_rate,*,auditor,soft_sync_boundaries=False):
    """Accept only a fully reaudited translation of an otherwise refused clip.

    Both edges must be covered by soft SYNC overlaps. Missing observations,
    varying drift that cannot fit the same 3 ms clock, and unbounded offsets
    remain refusals. Passing old renders are byte-identical and never enter.
    """
    if (warp.get('constant_output_phase',{}).get('applied')or warp.get('fine_attack_timing')
            or warp.get('native_attack_preservation')):return audit
    points=audit.get('measured',[])
    if not soft_sync_boundaries or not points or not _coverage(audit,warp):return audit
    delays=[p['delay_seconds']for p in points]
    scale=max(abs(p.get('expected_seconds',0.))for p in points)
    if not np.isfinite(delays).all():raise ValueError('Nonfinite phase observations')
    maximum=max(map(abs,delays));tolerance=.003
    if not _timing_limit_exceeded(maximum,tolerance,reference_seconds=scale):return audit
    # One detector sample of headroom; tolerance itself remains exactly 3 ms.
    margin=1/8000
    lower=max(delays)-tolerance+margin;upper=min(delays)+tolerance-margin
    if lower>upper:return audit
    frames=math.ceil(lower*sample_rate)if lower>0 else math.floor(upper*sample_rate)if upper<0 else 0
    if not frames or abs(frames)>math.floor(.010*sample_rate):return audit
    path=Path(path);candidate=path.with_name(path.stem+'-phase-'+uuid4().hex+'.wav')
    keys=lambda a:{(p['role'],p['beat'],p.get('view'))for p in a['measured']}
    try:
        info=sf.info(path)
        if info.samplerate!=sample_rate:raise ValueError('Phase input has a different sample rate')
        shift_stream(path,candidate,int(frames))
        checked=auditor(candidate,warp,sample_rate)
        after=checked.get('maximum_measured_drift_seconds')
        if (after is None or _timing_limit_exceeded(after,tolerance,reference_seconds=scale)
                or not keys(audit)<=keys(checked)or not _coverage(checked,warp)):
            return audit
        if sf.info(candidate).frames!=info.frames:raise ValueError('Phase correction changed duration')
        candidate.replace(path)
        warp['constant_output_phase']={'applied':True,'advance_frames':int(frames),'sample_rate':sample_rate,
            'maximum_before_seconds':maximum,'maximum_after_seconds':after,'tolerance_seconds':tolerance,
            'all_measured_observations_retained':True,'mandatory_views_rechecked':True,
            'boundary_scope':'both edges inside mandatory soft SYNC overlaps',
            'padded_edge_frames':abs(int(frames)),'playback_rate':1.,'pitch_changed':False,
            'resampling':False,'isolated_attacks_copied':False,
            'method':'one whole-clip integer sample translation; actual PCM reaudited on the unchanged shared clock'}
        return checked
    finally:candidate.unlink(missing_ok=True)
