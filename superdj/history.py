"""Private completed-set history; exact recordings and normalized track names."""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,os,re,time,unicodedata
from .audio import AUDIO_EXTENSIONS

def track_key(path):
    name=Path(path).name
    while Path(name).suffix.lower() in AUDIO_EXTENSIONS:
        name=Path(name).stem
    name=re.sub(r'^unknown artist\s*-\s*','',name,flags=re.I)
    name=unicodedata.normalize('NFKD',name.casefold())
    return ''.join(c for c in name if c.isalnum() and not unicodedata.combining(c))

def history_path(path=None):
    if path is None:
        from .cues import runtime_config
        path=runtime_config().get('track_history')
    if not path:return None
    from .storage import private_path
    return private_path(path)

def read_history(path=None):
    path=history_path(path)
    empty={'schema_version':1,'sets':{},'sources':{}}
    if path is None or not path.is_file():return empty
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version')!=1:raise ValueError('Unsupported track history')
    return value

def import_cue_folder(folder,history=None):
    """The user's ready-set CUE files are explicit playback-history evidence."""
    from .storage import write_json
    target=history_path(history)
    if target is None:return {'imported_cues':0,'tracks':0}
    target.parent.mkdir(parents=True,exist_ok=True)
    entries=[]
    for cue in Path(folder).glob('**/*.cue'):
        raw=cue.read_bytes()
        if raw.startswith((b'\xff\xfe',b'\xfe\xff')):text=raw.decode('utf-16')
        else:
            try:text=raw.decode('utf-8-sig')
            except UnicodeDecodeError:text=raw.decode('cp1251')
        blocks=re.split(r'(?im)^\s*TRACK\s+\d+\s+AUDIO\s*$',text)[1:]
        titles=[]
        for block in blocks:
            title=re.search(r'(?im)^\s*TITLE\s+"([^"]*)"',block)
            performer=re.search(r'(?im)^\s*PERFORMER\s+"([^"]*)"',block)
            if not title:continue
            name=(performer.group(1)+' - ' if performer else '')+title.group(1)
            if not performer and ' - ' not in name and ' — ' not in name:continue
            titles.append(name)
        if titles:entries.append((cue,titles))
    with _locked(target):
        data=read_history(target)
        for cue,titles in entries:
            set_key='cue:'+str(cue.resolve()).casefold();ids=[]
            for title in titles:
                key=track_key(title+'.flac')
                digest='cue:'+hashlib.sha256(key.encode('utf-8')).hexdigest()
                row=data['sources'].setdefault(digest,{'path':None,'track_key':key,'source_bytes':None,'display_name':title,'identity_evidence':'user ready-set CUE artist/title'})
                row.setdefault('sets',[])
                if set_key not in row['sets']:row['sets'].append(set_key)
                ids.append(digest)
            data['sets'][set_key]={'cue_path':str(cue),'source_sha256':ids,'completed':True,'evidence':'user-designated ready set CUE'}
        write_json(target,data)
    return {'imported_cues':len(entries),'tracks':sum(len(t) for _,t in entries)}

def write_text_index(folder,history=None):
    data=read_history(history)
    directory=Path(folder)/'_Индекс_треков';directory.mkdir(parents=True,exist_ok=True)
    by_name={}
    for row in data['sources'].values():
        name=row.get('display_name') or (Path(row['path']).stem if row.get('path') else row['track_key'])
        entry=by_name.setdefault(row['track_key'],{'name':name,'file':row.get('path') or '', 'sets':set()})
        for identity in row.get('sets',[]):
            s=data['sets'].get(identity,{})
            path=s.get('audio_path') or s.get('cue_path')
            if path:entry['sets'].add(Path(path).stem)
    lines=['Использованные треки — индекс готовых сетов','Трек\tСеты\tИсходный файл (если известен)']
    for entry in sorted(by_name.values(),key=lambda v:v['name'].casefold()):
        lines.append(entry['name']+'\t'+'; '.join(sorted(entry['sets']))+'\t'+entry['file'])
    path=directory/'Использованные_треки.txt'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    return {'path':str(path),'unique_track_names':len(by_name)}

def content_hash(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def unseen_files(files,history=None):
    if history is None:
        from .cues import runtime_config
        ready=runtime_config().get('delivery_root')
        if ready:import_cue_folder(ready)
    h=read_history(history)
    sources=list(h['sources'].values())
    names={v['track_key'] for v in sources}
    sizes={v.get('source_bytes') for v in sources}
    hashes=set(h['sources'])
    unseen=[];excluded=[]
    for item in files:
        path=Path(item)
        reason=None
        if track_key(path) in names:reason='previously used artist/title/version name'
        elif path.stat().st_size in sizes and content_hash(path) in hashes:reason='previously used exact source content'
        if reason:excluded.append({'path':str(path),'reason':reason})
        else:unseen.append(path)
    return unseen,excluded

@contextmanager
def _locked(path):
    lock=path.with_suffix(path.suffix+'.lock');deadline=time.monotonic()+15
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,str(os.getpid()).encode());os.close(fd);break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('Track history is busy; retry after the active writer finishes')
            time.sleep(.1)
    try:yield
    finally:lock.unlink()

def register_manifest(manifest,history=None,manifest_path=None):
    from .storage import write_json
    target=history_path(history)
    if target is None:return {'registered':False,'reason':'private history not configured'}
    qa=dict(manifest.get('exports',{}).get('flac',{}))
    if manifest.get('delivery',{}).get('flac'):qa['path']=manifest['delivery']['flac']
    rows=manifest.get('selection_and_analysis',manifest.get('tracks',[]))
    if len(rows)<3 or not qa.get('full_decode_ok') or not Path(qa.get('path','')).is_file():
        return {'registered':False,'reason':'not a verified completed multi-track set'}
    prepared=[]
    for row in rows:
        path=Path(row.get('path',row.get('original_path','')))
        if not path.is_file():raise ValueError('Completed history source missing: '+str(path))
        digest=row.get('source_sha256') or row.get('provenance',{}).get('source_sha256') or content_hash(path)
        prepared.append((digest,{'path':str(path.resolve()),'track_key':track_key(path),'source_bytes':path.stat().st_size}))
    target.parent.mkdir(parents=True,exist_ok=True)
    set_key=str(Path(qa['path']).resolve()).casefold()
    with _locked(target):
        data=read_history(target)
        data['sets'][set_key]={'audio_path':qa['path'],'manifest_path':str(manifest_path) if manifest_path else None,
            'source_sha256':[d for d,_ in prepared],'completed':True}
        for digest,row in prepared:
            entry=data['sources'].setdefault(digest,row)
            entry.setdefault('sets',[])
            if set_key not in entry['sets']:entry['sets'].append(set_key)
        write_json(target,data)
    return {'registered':True,'tracks':len(prepared),'unique_sources':len(data['sources']),'sets':len(data['sets'])}
