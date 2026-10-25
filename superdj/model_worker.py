"""Private JSON-lines worker. Model objects stay alive until the run ends."""
import contextlib
import json
import os
import sys
import traceback


def _work():
    output = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                kind, job = request['kind'], request['job']
                if kind == 'cue':
                    from .cues import _worker
                    result = _worker(job)
                elif kind == 'stem':
                    from .stems import _worker
                    result = _worker(job)
                elif kind == 'beat':
                    import numpy as np
                    from pathlib import Path
                    from .analysis import _beat_this
                    # Model/tensor views may outlive inference in this persistent
                    # process. Own the array so the caller can remove its input
                    # immediately after the reply, including on Windows.
                    signal = np.load(job['signal'], allow_pickle=False)
                    beats, downbeats, evidence = _beat_this(signal, Path(job['checkpoint']), job['device'])
                    result = [beats.tolist(), downbeats.tolist(), evidence]
                elif kind == 'audio':
                    from .adapters import _worker
                    result = _worker(job)
                else:
                    raise ValueError('Unknown neural task')
            loaded = {}
            for name, attribute in (('cues', '_MODELS'), ('stems', '_MODELS'), ('analysis', '_TRACKERS')):
                module = sys.modules.get('superdj.' + name)
                loaded[name] = len(getattr(module, attribute, {})) if module else 0
            reply = {'result': result, 'loaded_models': loaded}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply = {'error': f'{type(error).__name__}: {error}'}
        from .resource_limits import current_report, current_settings
        if current_settings().enabled:
            # This is the actual interpreter, even when a venv launcher is its
            # parent. No neural module is imported before its own native scope.
            reply['resource_limits'] = {'pid': os.getpid(), 'report': current_report()}
        output.write(json.dumps(reply, allow_nan=False) + '\n')
        output.flush()




def main():
    from .resource_limits import current_settings,limit_scope
    # The parent serializes only SuperDJ limits into its owned worker's env.
    with limit_scope(current_settings(), native=True):
        return _work()


if __name__ == '__main__':
    main()
