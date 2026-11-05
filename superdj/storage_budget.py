"""Bounded generated audio storage. Originals and models are never eviction targets."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
import shutil
import stat
import time
import uuid
import tempfile

GIB = 1024 ** 3
DEFAULT_CACHE_GIB = 50
DEFAULT_WORK_GIB = 64
DEFAULT_TOTAL_GB = 64


class StorageLimitError(OSError):
    pass


def workstation_policy():
    from .resource_limits import runtime_path
    path=runtime_path()
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def validate_storage_path(path, *, policy=None):
    """Validate generated data before any mkdir, lock, log or audio write.

    Use the saved workstation policy even when a legacy SDK replaces its runtime
    dictionary. Resolving existing links also rejects junctions escaping the root.
    An explicit settings change may supply its candidate policy before commit.
    """
    path=Path(path).expanduser().resolve()
    policy=workstation_policy() if policy is None else policy
    base=policy.get('storage_root')
    if base:
        root=Path(base).expanduser().resolve()
        if path!=root and not path.is_relative_to(root):
            raise StorageLimitError(f'Generated storage must stay inside the configured working folder: {root}; rejected: {path}')
    for blocked in policy.get('forbidden_storage_roots',[]):
        root=Path(blocked).expanduser().resolve()
        if path==root or path.is_relative_to(root):
            raise StorageLimitError(f'Generated storage is prohibited here: {root}; use the configured storage_root')


def temporary_parent():
    """Default generated scratch; never fall back to system temp after setup."""
    policy=workstation_policy()
    base=policy.get('storage_root')
    parent=Path(policy.get('scratch_root') or (Path(base)/'scratch' if base else Path(tempfile.gettempdir())/'superdj-scratch'))
    validate_storage_path(parent,policy=policy)
    return parent.expanduser().resolve()


def shared_workspace(*paths):
    policy=workstation_policy()
    for path in paths:
        if path is not None:validate_storage_path(path,policy=policy)
    base=Path(policy.get('storage_root') or policy.get('scratch_root') or Path(tempfile.gettempdir())/'superdj-scratch').expanduser().resolve()
    if any(p is not None and (Path(p).resolve()==base or Path(p).resolve().is_relative_to(base)) for p in paths):
        value=policy.get('storage_max_gb',DEFAULT_TOTAL_GB)
        gib(value,'storage_max_gb')
        return base,int(value*1000**3)
    return None


def owned_roots(parent):
    """Only marked engine scratch, never arbitrary directories in this folder."""
    parent=Path(parent)
    if not parent.exists():
        return []
    result=[]
    for child in parent.iterdir():
        if child.is_dir() and not child.is_symlink():
            marker=child/'.superdj-run.json'
            if marker.is_file():
                value=json.loads(marker.read_text(encoding='utf8'))
                if value.get('schema')=='superdj-temporary-v1':
                    result.append((child,int(value['pid'])))
    return result


def cleanup_stale(parent):
    from .resource_config import _lock
    validate_storage_path(parent)
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    with _lock(parent/'.cleanup.lock'):
        for child,pid in owned_roots(parent):
            if not _alive(pid):
                _remove(parent,child)


def temporary(parent,prefix):
    cleanup_stale(parent)
    value=tempfile.TemporaryDirectory(prefix=prefix,dir=parent)
    pending=Path(value.name)/'.owner-pending'
    pending.write_text(json.dumps({'schema':'superdj-temporary-v1','pid':os.getpid()}),encoding='utf8')
    os.replace(pending,Path(value.name)/'.superdj-run.json')
    return value


def gib(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number of GiB')
    return int(value * GIB)


def tree_bytes(root):
    """Metadata only, bounded to the specified owned tree; no directory symlinks."""
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    started = time.monotonic()
    stack = [root]
    count = 0
    while stack:
        path = stack.pop()
        with os.scandir(path) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                tag = getattr(info, 'st_reparse_tag', 0)
                cloud = tag and (tag & 0xffff0fff) == 0x9000001a
                if entry.is_symlink() or (tag and not cloud):
                    raise StorageLimitError('Storage tree contains a link; choose a dedicated physical folder')
                if stat.S_ISDIR(info.st_mode):
                    stack.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                count += 1
                if count > 200000 or time.monotonic()-started > 30:
                    raise StorageLimitError('Generated-storage tree contains too many entries; clean it before continuing')
    return total


def _remove(root, path):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink() or path.resolve() == root or not path.resolve().is_relative_to(root):
        raise StorageLimitError('Unsafe cache deletion target rejected')
    # Prove nested links cannot redirect deletion outside this dedicated cache.
    tree_bytes(path)
    shutil.rmtree(path)


def _alive(pid):
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        dll=ctypes.WinDLL('kernel32',use_last_error=True)
        dll.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        dll.OpenProcess.restype=wintypes.HANDLE
        dll.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        dll.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=dll.OpenProcess(0x1000,False,pid)  # query only; never signal a process
        if not handle:
            return ctypes.get_last_error()==5  # inaccessible means protected
        try:
            code=wintypes.DWORD()
            return not dll.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value==259
        finally:
            dll.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


class CacheBudget:
    def __init__(self, root, maximum):
        validate_storage_path(root)
        self.root = Path(root).resolve()
        allowed={'.storage','objects-v2','objects-v1','analysis','cue-detr-jobs','energy-flow-v1',
                 'final-percussion','ri1','scratch','stem-locks','vocals','models','README.txt'}
        if self.root.exists():
            foreign=[p.name for p in self.root.iterdir() if p.name not in allowed and not p.name.startswith('analysis-')]
            if foreign:
                raise StorageLimitError('Choose a dedicated cache folder; existing unrelated data must not become eviction targets')
        self.maximum = int(maximum)
        if self.maximum <= 0:
            raise ValueError('Persistent cache needs a positive byte limit')
        self.control = self.root / '.storage'
        self.control.mkdir(parents=True, exist_ok=True)
        self.lease = self.control / 'leases' / f'{os.getpid()}-{uuid.uuid4().hex}'
        with self.locked():
            self.lease.mkdir(parents=True)
            (self.lease / 'owner.json').write_text(json.dumps({'pid': os.getpid(), 'created': time.time()}), encoding='utf8')
        self.evicted_bytes = 0
        self.skipped_writes = 0
        shared=shared_workspace(self.root)
        self.shared_budget=WorkBudget(shared[1],[shared[0]]) if shared else None
        if shared:self.maximum=min(self.maximum,shared[1])

    @contextmanager
    def locked(self):
        from .resource_config import _lock
        with _lock(self.control / 'quota.lock', timeout=120.):
            yield

    def pin(self, path):
        relative = str(Path(path).resolve().relative_to(self.root))
        with (self.lease / 'pins.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(relative) + '\n')
        if Path(path).is_dir():
            (Path(path) / '.access').touch()

    def _pinned(self):
        result = set()
        for folder in (self.control / 'leases').iterdir():
            try:
                owner = json.loads((folder / 'owner.json').read_text(encoding='utf8'))
                if not _alive(int(owner['pid'])):
                    _remove(self.root, folder)
                    continue
                pins = folder / 'pins.jsonl'
                if pins.exists():
                    result.update(json.loads(line) for line in pins.read_text(encoding='utf8').splitlines() if line)
            except (OSError, ValueError, KeyError):
                # An unreadable live lease cannot authorize eviction.
                raise StorageLimitError('Cannot validate cache leases; persistent writes paused')
        return result

    def _entries(self):
        base = self.root / 'objects-v2'
        if not base.exists():
            return []
        entries = []
        for engine in base.iterdir():
            if not engine.is_dir() or engine.is_symlink():
                continue
            for stage in engine.iterdir():
                if not stage.is_dir() or stage.is_symlink():
                    continue
                for item in stage.iterdir():
                    if item.is_dir() and not item.is_symlink() and (item / 'manifest.json').is_file():
                        stamp = item / '.access'
                        entries.append((stamp.stat().st_mtime if stamp.exists() else item.stat().st_mtime, item, tree_bytes(item)))
        return entries

    def make_room(self, additional=0):
        """Caller owns quota.lock. Only complete, unleased stage objects are evicted."""
        used = tree_bytes(self.root)
        shared_used = self.shared_budget.usage() if self.shared_budget else 0
        def fits():
            return used+additional<=self.maximum and (not self.shared_budget or shared_used+additional<=self.shared_budget.maximum)
        if fits():
            return True
        pinned = self._pinned()
        for _, path, size in sorted(self._entries()):
            if fits():
                break
            if str(path.relative_to(self.root)) in pinned:
                continue
            _remove(self.root, path)
            used -= size
            if self.shared_budget:
                shared_used -= size
            self.evicted_bytes += size
        return fits()

    @contextmanager
    def admit(self, additional):
        with self.locked():
            allowed = additional <= self.maximum and self.make_room(additional)
            if not allowed:
                self.skipped_writes += 1
            yield allowed

    def close(self):
        with self.locked():
            _remove(self.root, self.lease)
            self.make_room()


class WorkBudget:
    def __init__(self, maximum, roots=()):
        self.maximum = int(maximum)
        if self.maximum <= 0:
            raise ValueError('Working storage needs a positive byte limit')
        self.roots = []
        self.peak = 0
        self.parents = []
        for root in roots:
            self.watch(root)

    def watch(self, root):
        validate_storage_path(root)
        root = Path(root).resolve()
        shared=shared_workspace(root)
        if shared:
            self.maximum=min(self.maximum,shared[1])
            root=shared[0]  # Direct SDK budgets also count the complete folder.
        if not any(root == parent or root.is_relative_to(parent) for parent in self.roots):
            self.roots = [p for p in self.roots if not p.is_relative_to(root)] + [root]

    def watch_parent(self,parent):
        parent=Path(parent).resolve()
        if parent not in self.parents:
            self.parents.append(parent)
        self.watch(parent)

    def usage(self):
        roots=list(self.roots)
        for parent in self.parents:
            roots.extend(root for root,_ in owned_roots(parent))
        unique=[]
        for root in sorted(set(roots),key=lambda p:len(p.parts)):
            if not any(root.is_relative_to(p) for p in unique):unique.append(root)
        used = sum(tree_bytes(root) for root in unique)
        self.peak = max(self.peak, used)
        return used

    def check(self, additional=0):
        used=self.usage()
        if used + additional > self.maximum:
            raise StorageLimitError(f'Working-storage limit exceeded: {(used+additional)/GIB:.2f} GiB required, '
                                    f'{self.maximum/GIB:g} GiB allowed. Increase --scratch-max-gib or use a smaller set; audio quality is unchanged.')
        return used


def limits(runtime=None):
    if runtime is None:
        from .cues import runtime_config
        runtime = runtime_config()
    return (gib(runtime.get('cache_max_gib', DEFAULT_CACHE_GIB), 'cache_max_gib'),
            gib(runtime.get('scratch_max_gib', DEFAULT_WORK_GIB), 'scratch_max_gib'))


def configure(action='status', *, path=None, cache_root=None, cache_max_gib=None,
              scratch_root=None, scratch_max_gib=None, storage_root=None, storage_max_gb=None):
    from .resource_limits import runtime_path
    from .resource_config import _lock, _read, _atomic
    target = Path(path) if path is not None else runtime_path()
    with _lock(target.with_name('storage-settings.lock')):
        raw = _read(target)
        value = json.loads(raw.decode('utf-8-sig')) if raw else {}
        if action not in ('status', 'configure', 'off'):
            raise ValueError('Use storage status, configure or off')
        if action != 'status':
            if storage_root is not None:
                from .storage import private_path
                base=private_path(storage_root)
                candidate={**workstation_policy(),**value,'storage_root':str(base)}
                validate_storage_path(base,policy=candidate)
                value.update(storage_root=str(base),cache_root=str(base/'cache'),scratch_root=str(base/'scratch'),private_output_root=str(base/'reports'))
            if storage_max_gb is not None:
                gib(storage_max_gb,'storage_max_gb')
                value['storage_max_gb']=storage_max_gb
            if cache_max_gib is not None:
                gib(cache_max_gib, 'cache_max_gib'); value['cache_max_gib'] = cache_max_gib
            if scratch_max_gib is not None:
                gib(scratch_max_gib, 'scratch_max_gib'); value['scratch_max_gib'] = scratch_max_gib
            for name, supplied in (('cache_root', cache_root), ('scratch_root', scratch_root)):
                if supplied is not None:
                    from .storage import private_path
                    validate_storage_path(supplied,policy={**workstation_policy(),**value})
                    value[name] = str(private_path(supplied))
            if action == 'off':
                value['cache_mode'] = 'off'
            elif cache_root is not None:
                value['cache_mode'] = 'persistent'
            for name in ('cache_root','scratch_root','private_output_root'):
                if value.get(name):validate_storage_path(value[name],policy={**workstation_policy(),**value})
            value['storage_configured'] = True
            limits(value)
            _atomic(target, value, expected=raw, compare=True)
    import tempfile
    return {'storage_root':value.get('storage_root'),'storage_max_gb':value.get('storage_max_gb',DEFAULT_TOTAL_GB),
            'cache_mode': value.get('cache_mode', 'off'), 'cache_root': value.get('cache_root'),
            'cache_max_gib': value.get('cache_max_gib', DEFAULT_CACHE_GIB),
            'scratch_root': value.get('scratch_root', str(Path(tempfile.gettempdir())/'superdj-scratch')),
            'scratch_max_gib': value.get('scratch_max_gib', DEFAULT_WORK_GIB),
            'storage_configured': value.get('storage_configured', False)}


def setup(*, path=None, ask=input):
    """An explicit first-run choice; Enter keeps persistent audio caching off."""
    current=configure(path=path)
    local=Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.cache')))/'SuperDJ'
    base=ask(f'Working folder [{current.get("storage_root") or local}]: ').strip() or str(current.get('storage_root') or local)
    total=float(ask(f'Total storage limit GB [{current["storage_max_gb"]}]: ').strip() or current['storage_max_gb'])
    enabled=ask('Keep processed audio between runs? [y/N]: ').strip().lower() in ('y','yes','да')
    cache_root=None
    cache_limit=current['cache_max_gib']
    if enabled:
        cache_root=ask(f'Cache folder [{Path(base)/"cache"}]: ').strip() or str(Path(base)/'cache')
        cache_limit=float(ask(f'Cache limit GiB [{cache_limit}]: ').strip() or cache_limit)
    scratch=str(Path(base)/'scratch')
    work_limit=current['scratch_max_gib']
    return configure('configure' if enabled else 'off',path=path,cache_root=cache_root,
                     cache_max_gib=cache_limit,scratch_root=scratch,scratch_max_gib=work_limit,
                     storage_root=base,storage_max_gb=total)
