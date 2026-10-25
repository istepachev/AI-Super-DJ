"""Native scheduling for the calling CLI or a known owned child; no scanner."""
import ctypes
import os


def selected_cpus(allowed, maximum):
    values = sorted(set(allowed))
    if not values:
        raise ValueError('No available process CPUs were reported')
    return values[:maximum]


def _windows(process, limits):
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    native = ctypes.WinDLL('ntdll')
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetPriorityClass.restype = wintypes.BOOL
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.GetPriorityClass.restype = wintypes.DWORD
    native.NtSetInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.ULONG]
    native.NtSetInformationProcess.restype = wintypes.LONG
    native.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, wintypes.ULONG,
                                                ctypes.c_void_p]
    native.NtQueryInformationProcess.restype = wintypes.LONG
    # Popen retains the actual child handle, avoiding process-ID recycling.
    handle = kernel.GetCurrentProcess() if process is None else int(process._handle)
    report = {'platform': 'windows', 'errors': [],
              'scope': 'current processor group; inherited CPU mask is never expanded'}
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
        allowed = [i for i in range(ctypes.sizeof(current)*8) if current.value & (1 << i)]
        chosen = selected_cpus(allowed, limits.cpu_cores)
        mask = sum(1 << i for i in chosen)
        if not kernel.SetProcessAffinityMask(handle, mask):
            report['errors'].append('SetProcessAffinityMask failed')
        if kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
            report['cpu_ids'] = [i for i in allowed if current.value & (1 << i)]
        else:
            report['errors'].append('Affinity readback failed')
    else:
        report['errors'].append('GetProcessAffinityMask failed')
    if not kernel.SetPriorityClass(handle, 0x40):
        report['errors'].append('SetPriorityClass(Idle) failed')
    report['priority_class'] = int(kernel.GetPriorityClass(handle))
    if not report['priority_class']:
        report['errors'].append('Priority class readback failed')
    priority = wintypes.ULONG(0)
    status = native.NtSetInformationProcess(handle, 33, ctypes.byref(priority), ctypes.sizeof(priority))
    if status < 0:
        report['errors'].append(f'Process I/O priority unavailable: NTSTATUS 0x{status & 0xffffffff:08x}')
    queried = native.NtQueryInformationProcess(handle, 33, ctypes.byref(priority),
                                              ctypes.sizeof(priority), None)
    report['io_priority'] = int(priority.value) if queried >= 0 else None
    if queried < 0:
        report['errors'].append('I/O priority readback unavailable')
    return report


def apply_owned_limits(limits, process=None):
    """Limit self or a Popen child, never an arbitrary PID or other project.

    No memory trimming, process termination, job-object attachment, startup
    registration or periodic monitor is performed. Disabling the profile does
    not widen limits imposed by another application or an external guard.
    """
    if not limits.enabled:
        return {'applied': False, 'reason': 'economy profile disabled'}
    if process is not None:
        if not hasattr(process, 'poll') or not hasattr(process, 'pid'):
            raise TypeError('A known Popen child is required, not a raw PID')
        if process.poll() is not None:
            return {'applied': False, 'reason': 'owned child already finished'}
        if os.name != 'nt':
            # CPU/nice APIs target numeric PIDs on these platforms. Avoid a
            # child-exit/PID-reuse race; the actual model worker applies its
            # own native policy before importing neural libraries.
            return {'applied': False, 'reason': 'owned worker applies policy to itself; no child PID targeted',
                    'worker_self_policy_pending': True}
    try:
        if os.name == 'nt':
            report = _windows(process, limits)
        else:
            pid = 0 if process is None else process.pid
            report = {'platform': os.name, 'errors': [], 'io_priority': None}
            if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
                allowed = os.sched_getaffinity(pid)
                os.sched_setaffinity(pid, selected_cpus(allowed, limits.cpu_cores))
                report['cpu_ids'] = sorted(os.sched_getaffinity(pid))
            else:
                report['affinity_supported'] = False
            if hasattr(os, 'setpriority'):
                os.setpriority(os.PRIO_PROCESS, pid, 19)
                report['nice'] = os.getpriority(os.PRIO_PROCESS, pid)
            report['io_priority_status'] = 'not configured on this platform'
        report['complete_policy_applied'] = bool(os.name == 'nt' and not report.get('errors')
            and report.get('cpu_ids') and len(report['cpu_ids']) <= limits.cpu_cores
            and report.get('priority_class') == 0x40 and report.get('io_priority') == 0)
        report['applied'] = report['complete_policy_applied']
        report['partial'] = not report['complete_policy_applied'] and bool(
            report.get('cpu_ids') or report.get('priority_class') or report.get('nice') is not None)
        return report
    except (OSError, AttributeError, ValueError) as error:
        return {'applied': False, 'error': f'{type(error).__name__}: {error}',
                'queue_and_math_limits_retained': True}
