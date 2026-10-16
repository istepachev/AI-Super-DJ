"""Private JSON-lines worker. Model objects stay alive until the run ends."""
import contextlib
import json
import sys
import traceback


def main():
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
        output.write(json.dumps(reply, allow_nan=False) + '\n')
        output.flush()


if __name__ == '__main__':
    main()
