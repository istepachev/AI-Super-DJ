"""Keep native source coordinates separate from a prepared short-bar repeat."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .render import RenderError


def recipe(edit):
    """Accept only bounded edits with an established native inverse."""
    legacy=edit.get('method')=='one short stable whole-bar repeat inside solo body'
    modern=edit.get('method')=='verified phrase repeat with complementary-band join'
    if not legacy and not modern:
        raise RenderError('Unknown source timeline edit; analyze the original again')
    start, insertion, length = (float(edit[k]) for k in
        ('loop_start_seconds', 'insertion_source_seconds', 'added_seconds'))
    original = float(edit['original_source_duration_seconds'])
    count=1 if legacy else edit.get('repeat_count')
    rate=edit.get('sample_rate',48000) if modern else 48000
    if isinstance(rate,bool) or not isinstance(rate,int) or not 1000<=rate<=384000:
        raise RenderError('Invalid sample rate in source timeline edit')
    if (isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=4
            or not np.isfinite([start,insertion,length,original]).all() or start < 0
            or not 0 < length <= (10 if legacy else 60)
            or abs(insertion-start-length/count)>(1e-6 if legacy else 1/rate+1e-6)
            or insertion >= original or edit.get('bars_repeated') not in ((1,2,4) if legacy else (1,2,4,8,16))):
        raise RenderError('Invalid source timeline edit; analyze the original again')
    return start, insertion, length, original


def native_time(value, edit):
    start, insertion, length, _ = recipe(edit)
    if value < insertion-1e-6:
        return value
    if value < insertion+length-1e-6:
        cycle=length/edit.get('repeat_count',1)
        return start+(value-insertion)%cycle
    return value-length


def native_records(records):
    rows=deepcopy(records)
    for row in rows:
        edit=row.get('timeline_edit')
        if not edit:
            continue
        _, insertion, length, original=recipe(edit)
        if abs(float(row['duration'])-original-length)>1e-6:
            raise RenderError('Edited record duration contradicts its source timeline')
        virtual_beats=np.asarray(row['beats'],dtype=float)
        keep=(virtual_beats<insertion-1e-6)|(virtual_beats>=insertion+length-1e-6)
        for key in ('beats','downbeats'):
            values=np.asarray(row[key],dtype=float)
            values=values[(values<insertion-1e-6)|(values>=insertion+length-1e-6)]
            row[key]=np.where(values>=insertion+length-1e-6,values-length,values).tolist()
        for key in ('neural_downbeat_probability_by_beat','neural_beat_probability_by_beat'):
            values=row.get('grid',{}).get(key,[])
            if len(values)==len(virtual_beats):
                row['grid'][key]=np.asarray(values)[keep].tolist()
            elif len(values) not in (0,int(keep.sum())):
                raise RenderError('Beat probabilities contradict the edited source timeline')
        for cue in row.get('cue_candidates',[]):
            if cue['time']>=insertion+length-1e-6:
                cue['time']-=length
                cue['seconds']=cue['time']
        structure=[]
        for item in row.get('structure',[]):
            if insertion-1e-6<=item['time']<insertion+length-1e-6:
                continue
            if item['time']>=insertion+length-1e-6:
                item['time']-=length
            structure.append(item)
        row['structure']=structure
        row['duration']=row['duration_seconds']=original
        row.pop('timeline_edit',None)
        row.pop('original_duration_seconds',None)
        row.get('grid',{}).pop('timeline_edit',None)
    return rows


def native_plan(plan):
    result=deepcopy(plan)
    for identity, section in result.get('sections',{}).items():
        edit=section.pop('minimum_repeat',None)
        if not edit:
            continue
        _, _, length, _=recipe(edit)
        for key in ('end','source_span_seconds','predicted_solo_seconds','predicted_play_seconds'):
            section[key]-=length
        for edge in result.get('edges',[]):
            if edge['from']==identity:
                edge['cue_locks']['outgoing_end_seconds']-=length
        result['section_warnings']=[w for w in result.get('section_warnings',[])
            if not w.startswith(identity+': optional smooth phrase loop')
            and (not w.startswith(identity+': one ') or 'stable-bar repeat meets minimum playback' not in w)]
    result.pop('minimum_playback_edits',None)
    result.pop('short_track_loop_reviews',None)
    return result


def native_prepared(prepared, directory):
    """Remove the inserted PCM only; retain originals and legacy native drums."""
    files=deepcopy(prepared)
    for identity, item in files.items():
        edit=item.get('provenance',{}).get('timeline_edit') if isinstance(item,dict) else None
        if not edit:
            continue
        start, insertion, length, _=recipe(edit)
        streams=edit.get('aligned_preparation_streams')
        # Before 0.20.7 only mixture and voice received the repeat.
        streams=streams if streams is not None else [k for k in ('path','vocal_path') if item.get(k)]
        if not streams or 'path' not in streams or any(k not in ('path','vocal_path','drum_path') or not item.get(k) for k in streams):
            raise RenderError('Unknown prepared streams in source timeline edit')
        target=Path(directory)/uuid4().hex
        target.mkdir(parents=True,exist_ok=False)
        for key in streams:
            destination=target/(identity+'_'+key+'.wav')
            with sf.SoundFile(item[key]) as source:
                if source.channels!=2 or source.subtype not in ('FLOAT','DOUBLE'):
                    raise RenderError('Timeline recovery requires stereo floating-point preparations')
                first=round(insertion*source.samplerate)
                count=round(length*source.samplerate)
                if first+count>=len(source):
                    raise RenderError('Timeline edit exceeds the prepared source')
                with sf.SoundFile(destination,'w',samplerate=source.samplerate,channels=2,subtype=source.subtype) as out:
                    context=(edit.get('native_join_contexts') or {}).get(key)
                    native_context=None
                    if context:
                        from .execution import sha256
                        if sha256(context['path'])!=context['sha256']:
                            raise RenderError('Native loop restoration context changed')
                        native_context=np.load(context['path'],allow_pickle=False)
                        if native_context.shape!=(context['frames'],2) or native_context.dtype!=np.float32 or not np.isfinite(native_context).all():
                            raise RenderError('Invalid native loop restoration context')
                    elif edit.get('method')=='verified phrase repeat with complementary-band join':
                        raise RenderError('Missing native context for a conditioned phrase loop')
                    remaining=first-(len(native_context) if native_context is not None else 0)
                    if remaining<0:raise RenderError('Native loop restoration context exceeds prefix')
                    while remaining:
                        block=source.read(min(262144,remaining),dtype='float64',always_2d=True)
                        out.write(block);remaining-=len(block)
                    if native_context is not None:out.write(native_context)
                    source.seek(first+count)
                    for block in source.blocks(blocksize=262144,dtype='float64',always_2d=True):
                        out.write(block)
            item[key]=str(destination)
        infos=[sf.info(item[k]) for k in ('path','vocal_path','drum_path') if item.get(k)]
        if len({(i.samplerate,i.frames,i.channels) for i in infos})!=1:
            raise RenderError('Restored preparations do not share the native sample timeline')
        item['provenance'].pop('timeline_edit')
        item['provenance']['removed_minimum_repeat']=deepcopy(edit)
    return files


def native_failure(error, records):
    """Retain raw rendered evidence; give window search separate native bans."""
    from .recovery import unsupported_attack_events
    identity=error.evidence.get('failed_track')
    edit=next((r.get('timeline_edit') for r in records if r['id']==identity),None)
    if edit and 'canonical_attack_rejections' not in error.evidence:
        roles=error.evidence.get('failed_roles',[])
        native={role:[native_time(t,edit) for t in unsupported_attack_events(error.evidence,role)] for role in roles}
        error.evidence['canonical_attack_rejections']=native
        error.evidence['failed_source_timeline_edit']=deepcopy(edit)
    return error


def render_with_minimum_recovery(plan,records,prepared,output,work,*,renderer=None,**controls):
    from .minimum import preserve_minimum
    from .recovery import render_with_recovery
    from .render import render_plan
    from .transition_quality import TransitionFailure
    renderer=renderer or render_plan
    rows=native_records(records)
    planned=native_plan(plan)
    files=native_prepared(prepared,Path(work)/'native_prepared')
    success={}
    def attempt(candidate, source_rows, source_files, destination, directory, **options):
        effective,edited,processed=preserve_minimum(candidate,source_rows,source_files,
            Path(directory)/('minimum_'+uuid4().hex))
        try:
            result=renderer(effective,edited,processed,destination,directory,**options)
        except TransitionFailure as error:
            raise native_failure(error,edited)
        success.update(plan=effective,records=edited,prepared=processed)
        return result
    rendered,_=render_with_recovery(planned,rows,files,output,work,renderer=attempt,**controls)
    return rendered,success['plan'],success['records'],success['prepared']
