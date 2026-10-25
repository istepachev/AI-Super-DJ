"""Conservative portable resource admission; never reduce audio precision."""
from pathlib import Path
import ctypes
import json
import os
import shutil
import subprocess
import tempfile
import sys

GIB = 1024**3
_AUTO=object()


def total_memory():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[
                (name,ctypes.c_ulonglong) for name in ('total','available','page_total',
                'page_available','virtual_total','virtual_available','extended')]
        value=Memory();value.length=ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return int(value.total)
    try:
        return int(os.sysconf('SC_PHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError,ValueError,OSError):
        return None


def memory_policy(*, memory=_AUTO, total=_AUTO):
    """Bound audio buffers, leave headroom for the OS and existing applications.

    The optional environment value is a downward-only run budget, not a claim
    that the OS has reserved or hard-limited this process tree.
    """
    memory=available_memory() if memory is _AUTO else memory
    total=total_memory() if total is _AUTO else total
    reserve=max(512*1024**2,min(2*GIB,int((total or 4*GIB)*.125)))
    if memory is not None: reserve=min(reserve,int(memory*.25))
    from .resource_limits import current_settings
    limits=current_settings()
    if limits.enabled:
        reserve=max(reserve,int(limits.memory_headroom_gib*GIB))
    budget=min(8*GIB,int(memory*.5),max(0,memory-reserve)) if memory is not None else 768*1024**2
    explicit=os.environ.get('SUPERDJ_MEMORY_BUDGET_MB')
    if explicit is not None:
        limit=int(explicit)*1024**2
        if limit<256*1024**2: raise ValueError('Memory budget must be at least 256 MiB')
        budget=min(budget,limit)
    constrained=memory is None or memory<4*GIB or budget<2*GIB
    return dict(available_memory_bytes=memory,total_memory_bytes=total,
                buffer_budget_bytes=budget,os_application_headroom_bytes=reserve,
                constrained=constrained,reservation=False)


def configure_math_threads():
    """Bound large native-library thread pools before importing audio libraries."""
    from .resource_limits import bootstrap_math_threads
    bootstrap_math_threads()
    if memory_policy()['constrained']:
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
            if name=='NUMBA_NUM_THREADS' and 'numba' in sys.modules:
                continue  # fixed launch capacity; execution scopes use the public thread mask
            os.environ[name]='1'


def available_memory():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                    'page_available', 'virtual_total', 'virtual_available', 'extended')]
        value = Memory(); value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return int(value.available)
    try:
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(values['MemAvailable'].split()[0])*1024
    except (OSError, KeyError, ValueError):
        try:
            return int(os.sysconf('SC_AVPHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
        except (AttributeError, ValueError, OSError):
            return None


def hardware():
    result = {'cpu_threads': os.cpu_count() or 1, 'available_memory_bytes': available_memory(),
              'total_memory_bytes':total_memory(),
              'cuda_probe': 'unknown', 'gpus': []}
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            value = subprocess.run([executable, '--query-gpu=index,memory.total,memory.free',
                '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5)
            output = value.stdout.decode('utf8') if isinstance(value.stdout,bytes) else value.stdout
            for line in output.splitlines():
                index, total, free = (int(item.strip()) for item in line.split(','))
                result['gpus'].append({'device': f'cuda:{index}', 'total_bytes': total*1024**2,
                                       'free_bytes': free*1024**2})
            result['cuda_probe'] = 'measured'
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return result


def _disk(path):
    directory = Path(path).expanduser().resolve()
    while not directory.exists():
        directory = directory.parent
    return directory, str(directory.stat().st_dev), shutil.disk_usage(directory).free


def admit_master(frames, sample_rate, channels=2, *, memory=None):
    """Select the actual complete master strategy before allocating audio."""
    if frames<=0 or sample_rate<=0 or channels not in (1,2):
        raise ValueError('Invalid mono/stereo complete master geometry')
    policy=memory_policy() if memory is None else memory_policy(memory=memory)
    # The upstream path duplicates mono during processing too.
    legacy=int(frames*2*8*6.5+GIB)
    bounded=legacy>policy['buffer_budget_bytes'] or policy['constrained']
    block=65536 if policy['constrained'] else 262144
    required=512*1024**2+block*2*8*20 if bounded else legacy
    if policy['buffer_budget_bytes']<required:
        raise MemoryError('Insufficient safe RAM for even the bounded complete master; '
                          'retain the verified premix and retry when memory is available')
    return {'frames':frames,'sample_rate':sample_rate,'required_bytes_estimate':required,
            'legacy_required_bytes_estimate':legacy,'mode':'bounded' if bounded else 'upstream',
            'block_frames':block,'temporary_pcm_bytes_estimate':int(frames*2*8*3+GIB) if bounded else 0,
            **policy}


def preflight(files, *, output, scratch=None, cache=None, jobs=None, cpu_jobs=None, devices=(), ffprobe='ffprobe',
              sample_rate=44100, target_play_seconds=240., machine=None, probe=None, disk=None):
    files = list(files)
    machine = hardware() if machine is None else machine
    disk = disk or _disk
    def inspect(path):
        value = subprocess.run([str(ffprobe), '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,duration:format=duration', '-of', 'json', str(path)],
            check=True, capture_output=True, timeout=30)
        data = json.loads(value.stdout); stream = data['streams'][0]
        return float(stream.get('duration') or data['format']['duration']), int(stream['sample_rate'])
    rows = [(probe or inspect)(path) for path in files]
    if not rows or any(not 0 < duration < float('inf') or not 0 < rate <= 768000 for duration, rate in rows):
        raise ValueError('Cannot estimate resources for invalid audio duration/rate')
    rate = max(sample_rate, max(row[1] for row in rows))
    seconds = sum(min(duration, target_play_seconds) for duration, _ in rows) if len(rows)>1 else rows[0][0]
    source_pcm = sum(duration*rate*2*4 for duration, _ in rows)
    rendered_pcm = seconds*rate*2*4
    legacy_master_peak = int(rendered_pcm*13 + GIB)
    task_peak = int(max(duration for duration, _ in rows)*rate*2*4*10 + 2*GIB)
    memory = machine.get('available_memory_bytes')
    policy=memory_policy(memory=memory,total=machine.get('total_memory_bytes'))
    master_bounded=legacy_master_peak>policy['buffer_budget_bytes'] or policy['constrained']
    master_peak=(512*1024**2+(65536 if policy['constrained'] else 262144)*2*8*20
                 if master_bounded else legacy_master_peak)
    reserve = policy['os_application_headroom_bytes']
    requested = min(4, max(1, int(jobs))) if jobs is not None else min(2, max(1, machine['cpu_threads']//4))
    requested_cpu = requested if cpu_jobs is None else min(4, max(1, int(cpu_jobs)))
    from .resource_limits import bounded_jobs,current_settings
    requested,requested_cpu=bounded_jobs(requested,requested_cpu)
    limits=current_settings()
    selected = requested
    if memory is not None:
        # A lower bound is a hard refusal; upper estimates cap concurrency and
        # remain explicit because OS memory pressure can change after admission.
        minimum=(768 if len(files)>1 else 320)*1024**2
        if policy['buffer_budget_bytes']<minimum:
            raise MemoryError('Insufficient safe RAM for bounded audio processing')
        selected = min(selected, max(1, int((memory-reserve)//task_peak)))
    selected = min(selected, max(1, machine['cpu_threads']//2))
    gpu_rows = {row['device']: row for row in machine.get('gpus', [])}
    chosen = list(devices)
    fallback=dict(missing=[],busy=[])
    if machine.get('cuda_probe') == 'measured':
        expanded = ['cuda:0' if item == 'cuda' else item for item in chosen]
        fallback['missing']=[item for item in expanded if item!='cpu' and item not in gpu_rows]
        fallback['busy']=[item for item in expanded if item in gpu_rows and gpu_rows[item]['free_bytes']<2*GIB]
        # Adapt to another computer or a busy GPU; preserve model and precision.
        chosen=[d for d in expanded if d=='cpu' or (d in gpu_rows and gpu_rows[d]['free_bytes']>=2*GIB)]
        if not chosen and expanded: chosen=['cpu']
        cuda_count = sum(device in gpu_rows for device in chosen)
        if cuda_count:
            selected = min(selected, cuda_count)
    # CPU preparation/rendering has its own queue. Reserve the concurrently
    # admitted analysis/model slots too, rather than multiplying the old RAM
    # allowance by two independent settings. Unknown RAM permits serial phases.
    if limits.enabled and memory is not None and memory-reserve < task_peak:
        raise MemoryError('Insufficient available RAM for the configured OS headroom and one full-quality audio task; retain checkpoints and retry when resources are available')
    capacity = max(1, int((memory-reserve)//task_peak)) if memory is not None else 1
    cpu_selected = min(requested_cpu, max(1, machine['cpu_threads']//2), max(1, capacity-selected))
    preparation_overlap = memory is not None and capacity >= selected+cpu_selected and not limits.enabled
    if policy['constrained']:
        selected=cpu_selected=1
        preparation_overlap=False
    allocations = [(scratch or Path(tempfile.gettempdir())/'superdj-scratch', int(source_pcm*4+rendered_pcm*10)),
                   (output, int(rendered_pcm*2))]
    allocations.append((cache or tempfile.gettempdir(), int(source_pcm*4+(rendered_pcm*3 if cache else 0))))
    volumes = {}
    for path, required in allocations:
        directory, identity, free = disk(path)
        entry = volumes.setdefault(identity, {'path': str(directory), 'free_bytes': free, 'required_bytes': GIB})
        entry['required_bytes'] += required
        entry['free_bytes'] = min(entry['free_bytes'], free)
    for entry in volumes.values():
        if entry['free_bytes'] < entry['required_bytes']:
            raise OSError(f"Insufficient temporary/output disk space at {entry['path']}: "
                f"{entry['required_bytes']/GIB:.1f} GiB estimated, {entry['free_bytes']/GIB:.1f} GiB free")
    return {'jobs': selected, 'requested_jobs': jobs, 'cpu_jobs': cpu_selected,
            'requested_cpu_jobs': cpu_jobs, 'preparation_overlap': preparation_overlap,
            'devices': chosen, 'hardware': machine,
            'memory_policy':policy,'short_lived_workers':policy['constrained'],
            'device_fallback':any(fallback.values()),'device_fallback_detail':fallback,
            'source_probes': [{'path':str(path),'duration_seconds':duration,'sample_rate':rate}
                              for path,(duration,rate) in zip(files,rows)],
            'estimated_master_peak_bytes': master_peak, 'estimated_task_peak_bytes': task_peak,
            'estimated_legacy_master_peak_bytes':legacy_master_peak,
            'master_mode':'bounded' if master_bounded else 'upstream',
            'memory_pressure_expected': memory is not None and master_peak+reserve>memory,
            'volumes': list(volumes.values()), 'estimate_is_reservation': False,
            'audio_precision_changed': False}
