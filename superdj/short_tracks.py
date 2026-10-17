"""One minute is the floor; two minutes is an optional extension objective."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import soundfile as sf

from .render import RenderError
from .loops import choose_repeat, repeat_file

METHOD='verified phrase repeat with complementary-band join'


def preserve_short_tracks(plan,records,prepared,directory):
    result,rows,files=deepcopy(plan),deepcopy(records),deepcopy(prepared)
    by_id={row['id']:row for row in rows};edits=[];reviews=[];directory=Path(directory)
    for position,identity in enumerate(result['order']):
        section=result.get('sections',{}).get(identity)
        if not section:continue
        # Older manifests used an engine-invented hard two-minute rule. It is
        # not a user cue lock and cannot survive as an accidental new minimum.
        section['minimum_play_seconds']=60.
        section['preferred_short_play_seconds']=120.
        played=float(section['predicted_play_seconds'])
        if played>=120:continue
        row=by_id[identity];item=files.get(identity)
        if not item or not item.get('path'):
            choice,reason=None,'no prepared PCM for an optional loop assessment'
        else:
            streams=[key for key in ('path','vocal_path','drum_path') if item.get(key)]
            infos={key:sf.info(item[key]) for key in streams};native=infos['path']
            if any((i.samplerate,i.frames,i.channels)!=(native.samplerate,native.frames,2)
                   or i.subtype not in {'FLOAT','DOUBLE'} for i in infos.values()):
                raise RenderError('A minimum repeat needs all prepared streams on the same stereo sample timeline')
            choice,reason=choose_repeat(result,row,position,section,item['path'])
        if not choice:
            review={'id':identity,'native_play_seconds':played,'loop_applied':False,'reason':reason,
                    'short_play_retained':played>=60}
            reviews.append(review)
            if played<60-1e-6:
                raise RenderError(f'{identity}: selected playback is {played:.2f}s, below the one-minute floor; no supported loop, choose a longer source span or exclude this short source explicitly')
            result.setdefault('section_warnings',[]).append(f'{identity}: retains {played:.2f}s full playback; optional phrase loop unavailable ({reason})')
            continue
        edit={'id':identity,'method':METHOD,**choice,
            'aligned_preparation_streams':streams,'original_source_duration_seconds':row['duration'],
            'reason':'optional extension of short playback toward two minutes',
            'synthetic_beat_extension':False,'native_join_contexts':{},
            'repeat_selection':'supported stable phrase, low voice risk, native harmony/bass/level checks and actual mixed energy'}
        directory.mkdir(parents=True,exist_ok=True)
        for key in streams:
            destination=directory/(identity+'_loop_'+key+'.wav')
            context=directory/(identity+'_native_join_'+key+'.npy')
            edit['native_join_contexts'][key]=repeat_file(item[key],destination,edit,context_path=context)
            item[key]=str(destination)
        item.setdefault('provenance',{})['timeline_edit']=edit
        row['timeline_edit']=edit;row.setdefault('grid',{})['timeline_edit']=edit
        row['original_duration_seconds']=row['duration']
        added=edit['added_seconds'];length=edit['loop_seconds'];start=edit['loop_start_seconds'];insertion=edit['insertion_source_seconds'];count=edit['repeat_count']
        row['duration']+=added;row['duration_seconds']=row['duration']
        native_beats=np.asarray(row['beats']);mask=(native_beats>=start-1e-6)&(native_beats<insertion-1e-6)
        for key in ('beats','downbeats'):
            values=np.asarray(row[key]);loop_values=values[(values>=start-1e-6)&(values<insertion-1e-6)]
            row[key]=np.sort(np.r_[np.where(values>=insertion-1e-6,values+added,values),
                                   *[loop_values+length*j for j in range(1,count+1)]]).tolist()
        for key in ('neural_downbeat_probability_by_beat','neural_beat_probability_by_beat'):
            values=np.asarray(row['grid'].get(key,[]))
            if len(values)==len(native_beats):
                ix=int(np.searchsorted(native_beats,insertion-1e-6))
                row['grid'][key]=np.r_[values[:ix],np.tile(values[mask],count),values[ix:]].tolist()
        for cue in row.get('cue_candidates',[]):
            if cue['time']>=insertion-1e-6:cue['time']+=added;cue['seconds']=cue['time']
        structure=deepcopy(row.get('structure',[]))
        loop_rows=[s for s in structure if start-1e-6<=s['time']<insertion-1e-6]
        row['structure']=sorted([{**s,'time':s['time']+(added if s['time']>=insertion-1e-6 else 0)} for s in structure]
            +[{**s,'time':s['time']+length*j} for j in range(1,count+1) for s in loop_rows],key=lambda s:s['time'])
        section['end']+=added
        for key in ('source_span_seconds','predicted_solo_seconds','predicted_play_seconds'):section[key]+=added
        section['minimum_repeat']=edit
        if position<len(result['edges']):result['edges'][position]['cue_locks']['outgoing_end_seconds']+=added
        result.setdefault('section_warnings',[]).append(f'{identity}: optional smooth phrase loop adds {added:.2f}s; full playback {section["predicted_play_seconds"]:.2f}s')
        edits.append(edit);reviews.append({'id':identity,'native_play_seconds':played,'loop_applied':True,
            'played_seconds':section['predicted_play_seconds'],'join_quality':edit['join_quality']})
    result.update(minimum_playback_edits=edits,short_track_loop_reviews=reviews,
        short_track_policy={'minimum_play_seconds':60.,'preferred_short_play_seconds':120.,
                            'loop_failure_keeps_native_short_play':True,'maximum_added_seconds':60.,'maximum_repeat_count':4})
    return result,rows,files
