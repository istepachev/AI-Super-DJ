"""Ready-folder delivery contains only music and its relative CUE."""
from pathlib import Path
import shutil
from .history import content_hash

def make_cue(manifest,path=None):
    audio=Path(manifest['exports']['flac']['path'])
    target=Path(path) if path else audio.with_suffix('.cue')
    names={r['id']:Path(r['path']).stem.removesuffix('.m4a') for r in manifest['selection_and_analysis']}
    quote=lambda v:'"'+v.replace('"',"'").replace('\n',' ').replace('\r',' ')+'"'
    lines=['PERFORMER "AI Super DJ"',f'TITLE {quote(audio.stem)}',f'FILE {quote(audio.name)} WAVE']
    for i,track in enumerate(manifest['render']['tracks'],1):
        artist,sep,title=names[track['id']].partition(' - ')
        if not sep:artist,title='Unknown Artist',artist
        ticks=round(track['start_seconds']*75);mins,rest=divmod(ticks,4500);secs,frames=divmod(rest,75)
        lines += [f'  TRACK {i:02} AUDIO',f'    TITLE {quote(title)}',f'    PERFORMER {quote(artist)}',f'    INDEX 01 {mins:02}:{secs:02}:{frames:02}']
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig',newline='\r\n')
    return target

def publish_pair(audio,cue,folder,*,replace=False,backup=None):
    audio,cue,folder=Path(audio),Path(cue),Path(folder).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    digest=content_hash(audio);destination=folder/audio.name
    suffix=2
    while destination.exists() and content_hash(destination)!=digest and not replace:
        destination=folder/f'{audio.stem}_v{suffix}{audio.suffix}';suffix+=1
    target_cue=destination.with_suffix('.cue')
    if destination.exists() and content_hash(destination)!=digest and replace:
        if backup is None:raise ValueError('Replacing a previous set requires a private backup directory')
        saved=Path(backup).resolve();saved.mkdir(parents=True,exist_ok=True)
        for original in (destination,target_cue):
            if original.exists():
                archived=saved/original.name
                if archived.exists():raise FileExistsError('Private backup already exists')
                shutil.copy2(original,archived)
                if content_hash(archived)!=content_hash(original):raise RuntimeError('Backup verification failed')
    if not destination.exists() or content_hash(destination)!=digest:
        pending=destination.with_suffix(destination.suffix+'.partial')
        shutil.copy2(audio,pending)
        if content_hash(pending)!=digest:raise RuntimeError('Delivery copy does not match the verified FLAC')
        pending.replace(destination)
    text=cue.read_text(encoding='utf-8-sig').replace(f'FILE "{audio.name}" WAVE',f'FILE "{destination.name}" WAVE')
    target_cue.write_text(text,encoding='utf-8-sig',newline='\r\n')
    return {'flac':str(destination),'cue':str(target_cue),'sha256':digest,'bit_exact_copy':True}
