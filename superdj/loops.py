"""Conservative, optional phrase loops with a reversible native-timeline edit.

The checks are engineering proxies, not a claim of inaudibility. A failed
candidate leaves the native short performance available to the caller.
"""
from pathlib import Path
import math
import numpy as np
import soundfile as sf

from .mixing import smooth, split_bands
from .render import RenderError, _grid, _beat_window, _local_window_evidence, _local_validation_possible


def _cosine(a,b):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denominator) if denominator>1e-15 else None


def _chroma(audio,sr):
    mono=audio.mean(axis=1);frequencies=np.fft.rfftfreq(len(mono),1/sr)
    power=abs(np.fft.rfft(mono*np.hanning(len(mono))))**2
    use=(frequencies>=100)&(frequencies<=min(3500,sr*.45))
    bins=np.rint(69+12*np.log2(frequencies[use]/440)).astype(int)%12
    return np.bincount(bins,weights=power[use],minlength=12)


def join_tail(outgoing,preroll,sample_rate,beat_seconds):
    """Complementary band cross before the next downbeat; no time removal.

    Input blocks end one sample before the two aligned source downbeats. The
    last sample therefore follows the native preroll into the copied bar.
    Apply this exact linear operator to mixture and all aligned estimates.
    """
    if outgoing.shape!=preroll.shape or outgoing.ndim!=2 or outgoing.shape[1]!=2:
        raise RenderError('Loop join requires matching stereo native context')
    if not np.isfinite(outgoing).all() or not np.isfinite(preroll).all():
        raise RenderError('Nonfinite native loop context')
    n=len(outgoing)
    if n<128:raise RenderError('Loop join has insufficient native context')
    progress=np.arange(n,dtype=float)
    bass_frames=min(n,max(2,round(beat_seconds*sample_rate)))
    bass=smooth((progress-(n-bass_frames))/(bass_frames-1))[:,None]
    broad=smooth(progress/(n-1))[:,None]
    result=sum(a*(1-g)+b*g for a,b,g in zip(split_bands(outgoing,sample_rate),split_bands(preroll,sample_rate),(bass,broad,broad)))
    # Exact endpoints preserve original neighbours even through filter roundoff.
    result[0]=outgoing[0];result[-1]=preroll[-1]
    return result.astype('float32')


def assess_join(outgoing,preroll,sr,beat_seconds):
    """Reject obvious harmonic/level conflicts and actual cancellation holes."""
    powers=[float(np.mean(x.astype(float)**2)) for x in (outgoing,preroll)]
    if min(powers)<1e-10:return None,'quiet native context does not establish a musical loop'
    level_db=abs(10*math.log10(powers[0]/powers[1]))
    if level_db>4:return None,'native phrase levels differ by more than 4 dB'
    chroma=_cosine(_chroma(outgoing,sr),_chroma(preroll,sr))
    if chroma is None or chroma<.92:return None,'native phrase harmony is insufficiently similar'
    # Retain stereo phase: a mono fold-down can cancel a real bass part.
    lows=[split_bands(x,sr)[0].reshape(-1) for x in (outgoing,preroll)]
    bass_active=min(float(np.mean(x*x)) for x in lows)>max(powers)*.01
    bass_similarity=_cosine(*lows) if bass_active else None
    if bass_active and (bass_similarity is None or bass_similarity<.65):
        return None,'native bass waveforms do not support a clean cross'
    mixed=join_tail(outgoing,preroll,sr,beat_seconds)
    tile=max(1,round(.02*sr));holes={}
    views={'full':(outgoing,preroll,mixed),'bass':tuple(split_bands(x,sr)[0] for x in (outgoing,preroll,mixed))}
    for label,(a,b,combined) in views.items():
        longest=run=0
        for first in range(0,len(mixed)-tile+1,tile):
            area=slice(first,first+tile)
            available=max(float(np.mean(a[area]**2)),float(np.mean(b[area]**2)))
            hole=available>1e-10 and float(np.mean(combined[area]**2))<available*10**(-12/10)
            run=run+tile if hole else 0;longest=max(longest,run)
        holes[label]=longest/sr
    if max(holes.values())>.04:return None,'actual band cross creates an energy hole'
    return {'passed':True,'native_chroma_cosine':chroma,'native_level_difference_db':level_db,
        'native_bass_waveform_cosine':bass_similarity,'bass_view':'measured' if bass_active else 'not sufficiently active',
        'maximum_energy_hole_seconds':max(holes.values()),'energy_holes_seconds_by_view':holes,'band_hz':[180,3500],
        'bass_cross_seconds':min(len(outgoing)/sr,beat_seconds),'melodic_cross_seconds':len(outgoing)/sr,
        'limitations':'Native phrase similarity and PCM continuity proxies; not a subjective inaudibility guarantee'},None


def choose_repeat(plan,row,position,section,path,*,target_seconds=120.,maximum_added_seconds=60.):
    """Search actual stable bars, then assess a bounded number of PCM joins."""
    grid=_grid(row,.75)
    if not grid[0] and not _local_validation_possible(row,grid):return None,'no supported native grid for an optional loop'
    if row.get('protected_ranges') or section.get('locked_start') or section.get('locked_end'):
        return None,'source locks or protected material retained'
    beats,bars=grid[2:4];info=sf.info(path);sr=info.samplerate
    start=int(np.argmin(abs(beats-section['start'])));end=int(np.argmin(abs(beats-section['end'])))
    def count(edge):return edge.get('overlap_beats') or edge.get('overlap_bars',edge.get('section_overlap_bars',0))*4
    incoming=int(count(plan['edges'][position-1])) if position else 0
    outgoing=int(count(plan['edges'][position])) if position<len(plan['edges']) else 0
    if start+incoming>=len(beats) or end-outgoing<0:return None,'no interior body for a native phrase loop'
    core_start=float(beats[start+incoming]);core_end=float(beats[end-outgoing])
    voice=(row.get('vocal_stem') or {}).get('profile',[])
    choices=[]
    for end_index in np.flatnonzero((bars>core_start)&(bars<=core_end+1e-6))[::-1]:
        for bar_count in (16,8,4,2,1):
            if end_index<bar_count:continue
            try:
                window=_beat_window(beats,bars,int(end_index)-bar_count,bar_count,row['bpm'],sr)
                if not grid[0]:
                    evidence=_local_window_evidence(row,beats,window)
                    if not evidence['strong_local_neural_evidence'] or evidence['local_timing_confidence']<.75:continue
            except RenderError:continue
            intervals=np.diff(window.times);period=float(np.median(intervals))
            if np.max(abs(intervals/period-1))>.02:continue
            first,last=map(float,(window.times[0],window.times[-1]));length=last-first
            context=min(2*period,length/4)
            if first-context<core_start-1e-6 or length>maximum_added_seconds:continue
            risks=[float(v['foreground_risk']) for v in voice if first-context<=v['time']<last and 'foreground_risk' in v]
            risk=float(np.mean(risks)) if risks else None
            if risk is not None and risk>.35:continue
            repeat_count=min(4,max(1,round((target_seconds-section['predicted_play_seconds'])/length)))
            repeat_count=min(repeat_count,int(maximum_added_seconds//length))
            if repeat_count<1:continue
            added=length*repeat_count
            if section['predicted_play_seconds']+added<60-1e-6:continue
            score=abs(section['predicted_play_seconds']+added-target_seconds)/30+.12*(repeat_count-1)+(risk or 0)*4
            choices.append((score,-bar_count,-last,first,last,period,context,repeat_count,risk))
    for _,neg_bars,_,first,last,period,context,repeats,risk in sorted(choices)[:24]:
        frames=round(context*sr);insert=round(last*sr);begin=round(first*sr)
        with sf.SoundFile(path) as source:
            source.seek(insert-frames);tail=source.read(frames,dtype='float32',always_2d=True)
            source.seek(begin-frames);pre=source.read(frames,dtype='float32',always_2d=True)
        if tail.shape!=(frames,2) or pre.shape!=(frames,2):raise RenderError('Incomplete prepared native loop context')
        quality,reason=assess_join(tail,pre,sr,period)
        if quality:
            length=(insert-begin)/sr
            return {'loop_start_seconds':first,'insertion_source_seconds':last,'loop_seconds':length,
                'repeat_count':repeats,'added_seconds':length*repeats,'bars_repeated':-neg_bars,
                'join_frames':frames,'beat_seconds':period,'sample_rate':sr,'join_quality':quality,
                'estimated_voice_risk':risk},None
    return None,'no supported phrase join passed harmony, bass, level and actual energy checks'


def repeat_file(source,destination,edit,*,context_path):
    """Insert bounded copies, preserving untouched PCM and native end exactly."""
    from .execution import sha256
    destination=Path(destination);context_path=Path(context_path)
    if destination.exists() or context_path.exists():raise FileExistsError(destination)
    with sf.SoundFile(source) as inp:
        sr=inp.samplerate;first=round(edit['loop_start_seconds']*sr);last=round(edit['insertion_source_seconds']*sr)
        width=edit['join_frames'];count=edit['repeat_count']
        if sr!=edit['sample_rate'] or first<width or last<=first or last>=inp.frames:raise RenderError('Invalid prepared loop timeline')
        inp.seek(first);loop=inp.read(last-first,dtype='float32',always_2d=True)
        inp.seek(last-width);tail=inp.read(width,dtype='float32',always_2d=True)
        inp.seek(first-width);pre=inp.read(width,dtype='float32',always_2d=True)
        joined=join_tail(tail,pre,sr,edit['beat_seconds'])
        np.save(context_path,tail,allow_pickle=False)
        with sf.SoundFile(destination,'w',samplerate=sr,channels=2,subtype='FLOAT') as out:
            inp.seek(0);remaining=last-width
            while remaining:
                block=inp.read(min(262144,remaining),dtype='float32',always_2d=True)
                if not len(block):raise RenderError('Native loop insertion exceeds source')
                out.write(block);remaining-=len(block)
            out.write(joined)
            for copy in range(count):
                if copy<count-1:
                    out.write(loop[:-width]);out.write(joined)
                else:out.write(loop)
            inp.seek(last)
            for block in inp.blocks(blocksize=262144,dtype='float32',always_2d=True):out.write(block)
    return {'path':str(context_path),'sha256':sha256(context_path),'frames':width}
