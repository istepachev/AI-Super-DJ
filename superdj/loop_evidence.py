"""Runtime evidence on the verified edited preparation timeline; native inputs stay immutable."""
from copy import deepcopy
import hashlib,json
import soundfile as sf

def render_records(rows,prepared):
 if not any(row.get('timeline_edit')for row in rows):return rows,[]
 result=[deepcopy(row)if row.get('timeline_edit')else row for row in rows];redirects=[]
 for row in result:
  edit=row.get('timeline_edit');item=prepared.get(row['id'],prepared.get(str(row.get('path','')),{}))
  if not edit:continue
  signature=hashlib.sha256(json.dumps(edit,sort_keys=True).encode()).hexdigest()
  already_rebased=row.get('render_evidence_timeline_signature')==signature
  if not isinstance(item,dict) or not item.get('path') or not item.get('drum_path'):
   raise ValueError('Edited loop evidence needs its matching prepared drum stream')
  infos=[sf.info(item[k]) for k in ('path','vocal_path','drum_path') if item.get(k)]
  if len({(i.samplerate,i.frames,i.channels) for i in infos})!=1:
   raise ValueError('Edited mixture, voice and drums do not share one sample timeline')
  evidence=row.get('kick_evidence')
  if evidence:
   if not item.get('drum_path'):raise ValueError('Edited loop evidence needs its matching prepared drum stream')
   for role,value in evidence.items():
    old=deepcopy(value);value.update(path=item['drum_path'],source_start_seconds=0.)
    if old!=value:redirects.append({'id':row['id'],'role':role,'native_evidence':old,'edited_evidence':deepcopy(value),'same_verified_timeline_edit':edit['method']})
  # These are feature samples, copied by the same known timeline edit.
  profile=(row.get('vocal_stem')or{}).get('profile')
  if profile and not already_rebased:
   start,insertion,length,count=[edit[k]for k in ('loop_start_seconds','insertion_source_seconds','loop_seconds','repeat_count')]
   added=edit['added_seconds'];loop=[p for p in profile if start<=p['time']<insertion]
   row['vocal_stem']['profile']=sorted([{**p,'time':p['time']+(added if p['time']>=insertion else 0)}for p in profile]+
    [{**p,'time':p['time']+length*j}for j in range(1,count+1)for p in loop],key=lambda p:p['time'])
  row['render_evidence_timeline_signature']=signature
 return result,redirects
