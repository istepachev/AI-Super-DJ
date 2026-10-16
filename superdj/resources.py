"""Conservative portable resource admission; never reduce audio precision."""
from pathlib import Path
import ctypes
import json
import os
import shutil
import subprocess
import tempfile

GIB = 1024**3


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
    """Recheck the actual complete-set geometry immediately before its master."""
    memory = available_memory() if memory is None else memory
    required = int(frames*channels*8*6.5 + GIB)
    if memory is not None and memory<required:
        raise MemoryError(f'Insufficient available RAM for the complete master: '
            f'{required/GIB:.1f} GiB conservatively estimated, {memory/GIB:.1f} GiB available; '
            'verified cached preparation/rendering can be reused after freeing memory')
    return {'frames':frames,'sample_rate':sample_rate,'required_bytes_estimate':required,
            'available_memory_bytes':memory,'reservation':False}


def preflight(files, *, output, scratch=None, cache=None, jobs=None, devices=(), ffprobe='ffprobe',
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
    master_peak = int(rendered_pcm*10 + GIB)  # float64 master, limiter and measurement arrays
    task_peak = int(max(duration for duration, _ in rows)*rate*2*4*10 + 2*GIB)
    memory = machine.get('available_memory_bytes')
    reserve = GIB
    requested = min(4, max(1, int(jobs))) if jobs is not None else min(2, max(1, machine['cpu_threads']//4))
    selected = requested
    if memory is not None:
        # A lower bound is a hard refusal; upper estimates cap concurrency and
        # remain explicit because OS memory pressure can change after admission.
        minimum = int(rendered_pcm*3 + reserve)
        if memory < minimum:
            raise MemoryError(f'Insufficient available RAM for this complete-set master: '
                f'at least {minimum/GIB:.1f} GiB estimated, {memory/GIB:.1f} GiB available')
        selected = min(selected, max(1, int((memory-reserve)//task_peak)))
    selected = min(selected, max(1, machine['cpu_threads']//2))
    gpu_rows = {row['device']: row for row in machine.get('gpus', [])}
    chosen = list(devices)
    if machine.get('cuda_probe') == 'measured':
        expanded = ['cuda:0' if item == 'cuda' else item for item in chosen]
        missing = [item for item in expanded if item != 'cpu' and item not in gpu_rows]
        if missing:
            raise ValueError('Configured GPU is unavailable on this computer: '+', '.join(missing))
        chosen = expanded
        # Never send work to a busy device with less than the conservative
        # model allowance. A CPU fallback must be an explicit configuration.
        for device in chosen:
            if device in gpu_rows and gpu_rows[device]['free_bytes'] < 2*GIB:
                raise MemoryError(f'{device}: less than 2 GiB free VRAM; free the GPU or configure CPU')
        cuda_count = sum(device in gpu_rows for device in chosen)
        if cuda_count:
            selected = min(selected, cuda_count)
    allocations = [(scratch or Path(tempfile.gettempdir())/'superdj-scratch', int(source_pcm*4+rendered_pcm*7)),
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
    return {'jobs': selected, 'requested_jobs': jobs, 'devices': chosen, 'hardware': machine,
            'source_probes': [{'path':str(path),'duration_seconds':duration,'sample_rate':rate}
                              for path,(duration,rate) in zip(files,rows)],
            'estimated_master_peak_bytes': master_peak, 'estimated_task_peak_bytes': task_peak,
            'memory_pressure_expected': memory is not None and master_peak+reserve>memory,
            'volumes': list(volumes.values()), 'estimate_is_reservation': False,
            'audio_precision_changed': False}
