"""Verify a fresh reproduction of a retained phrase loop without changing it.

This optional revision tool does not choose cues or certify a musical mix.
The unchanged canonical preparation and edit identity remain usable after the
freshly restored/repeated PCM has been compared in full. In particular, WAV
header differences must not trigger a change to the feature-timeline signature.
"""
from copy import deepcopy
from pathlib import Path
import re
import numpy as np
import soundfile as sf

from .execution import sha256
from .loops import repeat_file
from .minimum_timeline import native_prepared
from .storage import write_json


class RetainedLoopVerificationError(ValueError):
    pass


def reprepare_retained_loop(identity, preparation, output):
    """Restore and repeat one modern loop, then compare every float64 sample.

    ``preparation`` is one prepared item, including its original timeline edit
    and saved native join contexts. ``output`` must be a new private directory.
    Source files and metadata are never overwritten. Available mixture, voice
    and drum streams must have identical stereo geometry. Failure retains the
    private diagnostics and raises; only complete PCM equality returns verified.

    The returned ``preparation`` deliberately retains the original paths and
    exact edit metadata, so an already shifted feature profile is not shifted
    again merely because a context backup has a different path. Fresh artifacts
    and their hashes are retained in the verification receipt. Keep canonical
    sources and contexts; this is not a cache relocation or a new loop recipe.
    """
    if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identity):
        raise ValueError('A retained preparation needs a safe recording identifier')
    if not isinstance(preparation, dict):
        raise ValueError('A retained preparation must be one prepared item')
    item = deepcopy(preparation)
    edit = (item.get('provenance') or {}).get('timeline_edit')
    if not edit or edit.get('method') != 'verified phrase repeat with complementary-band join':
        raise ValueError('Fresh retained-loop verification needs a reversible modern phrase loop')
    streams = [k for k in ('path', 'vocal_path', 'drum_path') if item.get(k)]
    if 'path' not in streams:
        raise ValueError('A retained loop needs its prepared mixture')
    infos = {k: sf.info(item[k]) for k in streams}
    geometry = {(i.samplerate, i.frames, i.channels) for i in infos.values()}
    if len(geometry) != 1 or any(i.channels != 2 or i.subtype not in {'FLOAT', 'DOUBLE'} for i in infos.values()):
        raise ValueError('Retained loop streams must share one stereo floating-point timeline')
    original_hashes = {k: sha256(item[k]) for k in streams}
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    receipt = directory / 'verification.json'
    report = {'schema_version': 1, 'id': identity, 'verified': False, 'streams': [],
              'timeline_edit_unchanged': True, 'musical_mix_verified': False}
    try:
        native = native_prepared({identity: item}, directory / 'native')[identity]
        for key in streams:
            target = directory / f'{identity}_fresh_{key}.wav'
            context = directory / f'{identity}_native_join_{key}.npy'
            context_info = repeat_file(native[key], target, edit, context_path=context)
            # Native restoration already checks the original context checksum.
            # Compare values as well, independently of a .npy header encoding.
            original_context = edit['native_join_contexts'][key]
            a = np.load(original_context['path'], allow_pickle=False)
            b = np.load(context, allow_pickle=False)
            if not np.isfinite(a).all() or not np.array_equal(a, b):
                raise RetainedLoopVerificationError(f'Fresh native join context differs in {key}')
            checked = 0
            with sf.SoundFile(item[key]) as original, sf.SoundFile(target) as fresh:
                if (original.samplerate, original.frames, original.channels) != (fresh.samplerate, fresh.frames, fresh.channels):
                    raise RetainedLoopVerificationError(f'Fresh retained-loop geometry differs in {key}')
                for block in original.blocks(blocksize=262144, dtype='float64', always_2d=True):
                    actual = fresh.read(len(block), dtype='float64', always_2d=True)
                    if not np.isfinite(block).all() or not np.array_equal(block, actual):
                        raise RetainedLoopVerificationError(f'Fresh retained-loop PCM differs in {key} at or after frame {checked}')
                    checked += len(block)
            if sha256(item[key]) != original_hashes[key]:
                raise RetainedLoopVerificationError(f'Canonical preparation changed during verification: {key}')
            report['streams'].append({'stream': key, 'frames_checked': checked,
                'canonical_path': str(item[key]), 'canonical_sha256': original_hashes[key],
                'fresh_path': str(target), 'fresh_sha256': sha256(target),
                'all_pcm_samples_identical': True, 'fresh_native_join_context': context_info,
                'original_native_join_context_sha256': original_context['sha256']})
        # A later stream must not hide a concurrent change to an earlier one.
        if any(sha256(item[k]) != original_hashes[k] for k in streams):
            raise RetainedLoopVerificationError('Canonical preparation changed during verification')
        report['verified'] = True
        report['canonical_content_reused_after_exact_pcm_comparison'] = True
        write_json(receipt, report)
        return {'preparation': item, 'verification': report, 'receipt': str(receipt)}
    except Exception as error:
        report['error'] = str(error)
        write_json(receipt, report)
        raise
