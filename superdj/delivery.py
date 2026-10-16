"""Ready-folder delivery contains only music and its relative CUE."""
from pathlib import Path
import shutil
import hashlib
import os
import uuid
from .history import content_hash


def copy_verified(source, destination, expected=None):
    """Hash source bytes during copying; independently reread the written copy."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
        raise ValueError('Verified copy must not overwrite its source')
    digest = hashlib.sha256()
    created = False
    try:
        with source.open('rb') as original, destination.open('xb') as output:
            created = True
            before = os.fstat(original.fileno())
            for block in iter(lambda: original.read(1024*1024), b''):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
            after = os.fstat(original.fileno())
        value = digest.hexdigest()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError('Source changed during verified copying')
        if (expected is not None and value != expected) or content_hash(destination) != value:
            raise RuntimeError('Delivery copy does not match the verified FLAC')
        shutil.copystat(source, destination)
        return value
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise

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
    digest=None;destination=folder/audio.name
    known={}
    def destination_matches(path):
        nonlocal digest
        if digest is None:
            digest=content_hash(audio)
        if path not in known:
            known[path]=content_hash(path)
        return known[path]==digest
    suffix=2
    while destination.exists() and not destination_matches(destination) and not replace:
        destination=folder/f'{audio.stem}_v{suffix}{audio.suffix}';suffix+=1
    target_cue=destination.with_suffix('.cue')
    if destination.exists() and not destination_matches(destination) and replace:
        if backup is None:raise ValueError('Replacing a previous set requires a private backup directory')
        saved=Path(backup).resolve();saved.mkdir(parents=True,exist_ok=True)
        for original in (destination,target_cue):
            if original.exists():
                archived=saved/original.name
                if archived.exists():raise FileExistsError('Private backup already exists')
                copy_verified(original,archived)
    if not destination.exists() or not destination_matches(destination):
        pending=destination.with_name(destination.name+'.partial-'+uuid.uuid4().hex)
        digest=copy_verified(audio,pending,expected=digest)
        pending.replace(destination)
    text=cue.read_text(encoding='utf-8-sig').replace(f'FILE "{audio.name}" WAVE',f'FILE "{destination.name}" WAVE')
    target_cue.write_text(text,encoding='utf-8-sig',newline='\r\n')
    return {'flac':str(destination),'cue':str(target_cue),'sha256':digest,'bit_exact_copy':True}
