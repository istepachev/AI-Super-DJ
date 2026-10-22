"""Recover a native clock before choosing sample cuts, without editing audio."""
from pathlib import Path
from copy import deepcopy
import tempfile
import numpy as np
import soundfile as sf


def calibrate_window(path, times, *, source_origin=0., band=None, ffmpeg='ffmpeg'):
    """Require a dominant, temporally distributed family in native source PCM.

    This supplies inferred beat positions, not invented attack observations.
    Variable pulses and isolated synthetic impulses retain the existing path.
    """
    from .audio import decode_wave_file
    from .kick_grid import attack_events, repeated_attacks, isolated_events
    from .timing_safety import _native_consensus
    times=np.asarray(times,dtype=float)
    with tempfile.TemporaryDirectory(prefix='superdj-native-clock-') as scratch:
        try:sf.info(path)
        except (RuntimeError,sf.LibsndfileError):
            decoded=Path(scratch)/'decoded.wav'
            decode_wave_file(path,decoded,sample_rate=44100,ffmpeg=ffmpeg)
            path=decoded
        with sf.SoundFile(path) as stream:
            first=max(0,round((times[0]-source_origin-.2)*stream.samplerate))
            last=min(stream.frames,round((times[-1]-source_origin+.2)*stream.samplerate))
            stream.seek(first);audio=stream.read(last-first,dtype='float32',always_2d=True)
            base=source_origin+first/stream.samplerate;sr=stream.samplerate
        centres=times[1:-1]-base
        if len(centres)<8 or len(isolated_events(audio,sr,centres))>=.9*len(centres):return None
        chosen=tuple(band or (100,1400))
        rows=attack_events(audio,sr,centres,attack_band_hz=chosen)
        def family(observations):
            accepted=repeated_attacks(observations)
            relative=repeated_attacks([{**r,'time':r['time']-r['centre']} for r in observations],tolerance_seconds=.0125)
            return [observations[r['index']] for r in relative] if len(relative)>len(accepted) else accepted
        accepted=family(rows)
        if band is None and len(accepted)>=8:
            upper=family(attack_events(audio,sr,centres,attack_band_hz=(180,1200)))
            matched={r['index']:r for r in upper}
            offsets=[r['time']-matched[r['index']]['time'] for r in accepted if r['index'] in matched]
            if len(offsets)>=8 and .008<np.median(offsets)<.085 and np.percentile(offsets,90)-np.percentile(offsets,10)<.018:
                accepted=upper;chosen=(180,1200)
        clock=_native_consensus(accepted,float(np.median(np.diff(times))))
        if clock is None:return None
        calibrated=np.polyval(clock,np.arange(len(times))-1)+base
        error=calibrated-times
        if np.max(abs(error))>.12 or calibrated[0]<source_origin:return None
        with sf.SoundFile(path) as stream:
            if calibrated[-1]>source_origin+stream.frames/stream.samplerate:return None
        return {'times':calibrated.tolist(),'original_times':times.tolist(),
            'maximum_label_correction_seconds':float(np.max(abs(error))),
            'native_period_seconds':float(clock[0]),'reference_count':len(accepted),
            'attack_band_hz':list(chosen),'attack_observations_inferred':False,
            'method':'dominant native audio clock; coherent phase before sample cuts; original audio unchanged'}


def calibrate_render_windows(records, prepared, order, head, tail, sections, transitions, edges, *, ffmpeg, directory):
    """Retain explicit locks, overlap counts, output clocks and protected music."""
    from .execution import cached
    sections=deepcopy(sections);reports={}
    for position,identity in enumerate(order):
        record=records[identity];item=prepared.get(identity,prepared.get(str(record['path']),record['path']))
        item=item if isinstance(item,dict) else {'path':item}
        stem=record.get('vocal_stem') or {};evidence=record.get('kick_evidence') or {}
        section=sections.get(identity) or {};original={}
        for role,windows,edge_index,boundary,lock in (
                ('head',head,position-1,'start','incoming_start_seconds'),
                ('tail',tail,position,'end','outgoing_end_seconds')):
            window=windows[identity]
            if window is None or window.clock is None:continue
            source=evidence.get(role) or {}
            if source.get('allow_variable_attack_family'):continue
            path=source.get('path') or item.get('drum_path') or stem.get('drums_path') or item['path']
            origin=source.get('source_start_seconds',stem.get('drum_source_start_seconds',0.) if path!=item['path'] else 0.)
            result=cached('native_clock',{'path':Path(path),'times':window.times,'origin':origin,
                'band':source.get('attack_band_hz'),'ffmpeg':str(ffmpeg)},
                lambda:{'calibration':calibrate_window(path,window.times,source_origin=origin,band=source.get('attack_band_hz'),ffmpeg=ffmpeg)},directory)['calibration']
            if result is None:continue
            locked=section.get('locked_'+boundary)
            if locked is None:locked=(edges[edge_index].get('cue_locks') or {}).get(lock) is not None
            calibrated=np.asarray(result['times']);previous=window.times.copy()
            if locked:calibrated[0 if role=='head' else -1]=previous[0 if role=='head' else -1]
            start=calibrated[0] if role=='head' else section.get('start',0.)
            end=calibrated[-1] if role=='tail' else section.get('end',record.get('duration_seconds',record.get('duration')))
            if np.any(np.diff(calibrated)<=0) or any(start>float(r[0]) or end<float(r[1]) for r in record.get('protected_ranges',[])):continue
            original[role]=previous
            window.times=calibrated
            result={**result,'times':calibrated.tolist(),'boundary_locked':bool(locked)}
            reports.setdefault(identity,{})[role]=result
            transition=transitions[edge_index];prefix='incoming' if role=='head' else 'outgoing'
            transition[prefix+'_source_start']=float(calibrated[0]);transition[prefix+'_source_end']=float(calibrated[-1])
            transition[prefix+'_native_clock']=result
            if section:section[boundary]=float(calibrated[0 if role=='head' else -1])
        if original and section:
            body_delta=(float(tail[identity].times[0]-original['tail'][0]) if 'tail' in original else 0.)-(float(head[identity].times[-1]-original['head'][-1]) if 'head' in original else 0.)
            for key in ('predicted_solo_seconds','predicted_play_seconds','predicted_chapter_seconds'):
                if key in section:section[key]+=body_delta
            section['source_span_seconds']=section['end']-section['start']
            section['native_clock_calibration']=reports[identity]
    return sections,reports
