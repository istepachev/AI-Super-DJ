"""Reject duplicate full-deck correction work without truncating its search."""
from pathlib import Path
import shutil


def _signature(warp):
    # Match the exact control file consumed by the unchanged backend command.
    return (warp['output_frames'], ''.join(
        f"{p['source_frame']} {p['output_frame']}\n" for p in warp['anchors']))


def _stamp(path):
    try:
        stat = Path(path).stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return None


class CorrectionScreen:
    """Reuse only this clip's current PCM or its existing best-PCM snapshot.

    The source, decoder, gain and backend options are fixed by the owning clip
    render. No extra audio copies, persistent cache or cross-run state is used.
    Every reused candidate still receives the ordinary fresh acoustic audit.
    """
    def __init__(self):
        self.current = self.best = None
        self.same_map_skips = self.best_map_reuses = 0

    def rendered(self, warp, path):
        self.current = (_signature(warp), _stamp(path))

    def saved_best(self, warp, path):
        self.best = (_signature(warp), Path(path), _stamp(path))

    def reuse(self, warp, path):
        if (warp.get('source_attack_controls_unapplied') or
                warp.get('native_attack_preservation') or warp.get('fine_attack_timing')):
            # These operations can alter PCM independently of backend anchors.
            self.current = None
            return None
        key = _signature(warp)
        stamp = _stamp(path)
        if stamp is not None and self.current == (key, stamp):
            self.same_map_skips += 1
            return 'identical current backend map'
        if (self.best is not None and key == self.best[0]
                and self.best[2] is not None and _stamp(self.best[1]) == self.best[2]):
            shutil.copy2(self.best[1], path)
            self.rendered(warp, path)
            self.best_map_reuses += 1
            return 'identical already audited best backend map'
        return None

    def report(self):
        return {'same_map_backend_calls_skipped': self.same_map_skips,
                'best_map_backend_calls_reused': self.best_map_reuses,
                'correction_schedule_preserved': True,
                'fresh_acoustic_audit_required': True,
                'extra_audio_cache_files': 0}
