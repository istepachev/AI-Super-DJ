"""Independent PCM identity and preservation checks before delivery."""
import hashlib
import math
import numpy as np
import soundfile as sf


class WitnessWriter:
    def __init__(self, output):
        self.output = output
        self.digest = hashlib.sha256()
        self.frames = 0

    def write(self, audio):
        block = np.asarray(audio, dtype='<f4')
        if block.ndim != 2 or not np.isfinite(block).all():
            raise ValueError('Invalid assembled PCM block')
        self.digest.update(block.tobytes())
        self.frames += len(block)
        self.output.write(audio)


def verify_exported_pcm(master, encoded, *, gain_db=0.):
    """Every FLAC frame must match the master, allowing only declared quantization."""
    peak_error = 0.
    frames = 0
    with sf.SoundFile(master) as source, sf.SoundFile(encoded) as output:
        if (source.frames, source.samplerate, source.channels) != (output.frames, output.samplerate, output.channels):
            raise ValueError('Encoded PCM changed frame count, rate or channels')
        if output.format != 'FLAC' or output.subtype != 'PCM_24':
            raise ValueError('Delivery requires verified 24-bit FLAC')
        exact = source.subtype == 'PCM_24' and gain_db == 0
        gain = 10**(float(f'{gain_db:.4f}')/20)
        for block in source.blocks(blocksize=262144, dtype='float64', always_2d=True):
            actual = output.read(len(block), dtype='float64', always_2d=True)
            expected = block*gain
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError('Invalid PCM during export verification')
            error = float(np.max(np.abs(actual-expected)))
            if error > (0. if exact else 3/2**23):
                raise ValueError('Encoded PCM differs from the declared master; timing or sample data changed')
            peak_error = max(peak_error,error)
            frames += len(block)
    return {'verified':True,'frames':frames,'exact_pcm':exact,
            'maximum_sample_error':peak_error,'declared_gain_db':gain_db,
            'comparison':'every decoded sample against the master; quantization only when required'}


def verify_master_rhythm(reference, master, rendered, *, tolerance_seconds=.003):
    """Verify known shared-clock events survived mastering; uncertainty stops delivery.

    This preserves an already checked mix. It does not discover every possible
    percussion event or turn sparse recognition into a universal quality claim.
    """
    from scipy.signal import resample_poly, butter, sosfiltfilt, correlate
    reports=[]
    tracks={t['id']:t for t in rendered['tracks']}
    with sf.SoundFile(reference) as original, sf.SoundFile(master) as processed:
        if (original.frames,original.samplerate,original.channels)!=(processed.frames,processed.samplerate,processed.channels):
            raise ValueError('Mastering changed set timing or channels')
        sr=original.samplerate
        for edge in rendered['transitions']:
            if edge.get('tempo_technique')!='shared_phase_sync':
                continue
            events={}
            for identity,role in ((edge['from'],'tail'),(edge['to'],'head')):
                track=tracks[identity]
                audit=(track.get('warp') or {}).get('post_render_transient_audit',{}).get('final',{})
                if audit.get('maximum_measured_drift_seconds') is None or audit['maximum_measured_drift_seconds']>tolerance_seconds:
                    raise ValueError('Shared-phase transition lacks a passing acoustic audit')
                for point in audit.get('measured',[]):
                    if point['role']==role:
                        frame=round((track['set_start_seconds']+point['grid_output_seconds'])*sr)
                        events[frame]=max(events.get(frame,0.),abs(point['delay_seconds']))
            if not events:
                raise ValueError('No recognized attacks to verify after mastering')
            first=max(0,edge['set_start_frame']-sr)
            last=min(original.frames,edge['set_end_frame']+sr)
            original.seek(first);processed.seek(first)
            a=original.read(last-first,dtype='float64',always_2d=True)
            b=processed.read(last-first,dtype='float64',always_2d=True)
            divisor=math.gcd(sr,8000)
            a=resample_poly(a.mean(axis=1),8000//divisor,sr//divisor)
            b=resample_poly(b.mean(axis=1),8000//divisor,sr//divisor)
            measured=[];unresolved=[]
            bands=[(100,1400),(35,180)]
            filtered=[(sosfiltfilt(butter(3,band,btype='bandpass',fs=8000,output='sos'),a),
                       sosfiltfilt(butter(3,band,btype='bandpass',fs=8000,output='sos'),b)) for band in bands]
            for frame,prior_error in sorted(events.items()):
                center=round((frame-first)*8000/sr)
                choices=[]
                for band,(x,y) in zip(bands,filtered):
                    pre,post,search=160,640,40
                    lo,hi=center-pre,center+post
                    if lo-search<0 or hi+search>len(y):
                        continue
                    template=x[lo:hi]
                    region=y[lo-search:hi+search]
                    cross=correlate(region,template,mode='valid',method='direct')
                    power=np.convolve(region**2,np.ones(len(template)),mode='valid')
                    score=cross/np.sqrt(np.maximum(power*np.sum(template**2),1e-30))
                    peak=int(np.argmax(score));others=np.abs(np.arange(len(score))-peak)>8
                    margin=float(score[peak]-max(score[others])) if others.any() else 1.
                    if score[peak]>=.90 and margin>=.01:
                        choices.append({'delay_seconds':(peak-search)/8000,'correlation':float(score[peak]),'band_hz':list(band)})
                if not choices:
                    unresolved.append(frame)
                    continue
                if any(abs(p['delay_seconds'])+prior_error>tolerance_seconds for p in choices):
                    raise ValueError('Mastered percussion no longer meets the shared clock tolerance')
                measured.append({'frame':frame,'views':choices})
            if unresolved:
                raise ValueError('Mastered attacks remain ambiguous; cannot certify rhythm preservation')
            reports.append({'from':edge['from'],'to':edge['to'],'verified_events':len(measured),
                'maximum_additional_delay_seconds':max(abs(v['delay_seconds']) for p in measured for v in p['views']),
                'events':measured})
    return {'verified':True,'tolerance_seconds':tolerance_seconds,'transitions':reports,
            'scope':'preservation of recognized attacks in previously verified shared-phase overlaps'}
