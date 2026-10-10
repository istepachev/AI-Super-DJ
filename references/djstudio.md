# DJ.Studio adapter

AI Super DJ can inspect a native `.djs` archive, its exported CUE sheet and an
exported audio file. The audit adapter is read-only. An advanced preserving rewrite API
patches selected records in a new archive. It does not reproduce DJ.Studio's
synthesis or render its automation.

## Run an audit

```text
python -m superdj djstudio-audit --project "project.djs" --cue "export.cue" --audio "export.mp3" --work "private/report-folder"
```

`--project` and `--work` are required. `--cue` and `--audio` are optional.
Keep the report folder outside this repository. For FFmpeg tools outside PATH,
provide `--ffprobe "path/to/ffprobe"` and `--ffmpeg "path/to/ffmpeg"`.

- `--min-track-seconds 120` sets the requested minimum selected playback span.
- `--measure` decodes and measures the complete exported audio, including
  loudness and peaks. It requires `--audio`; without this option, audio is only
  probed for container duration, codec, sample rate and channels.
- `--include-native-state` includes a complete lossless JSON snapshot of
  `mixdata.json` and `settings.json` in the private report. This snapshot may
  contain personal file paths and should not be published.

The command writes `djstudio_audit.json`. Full measurement also writes an
`export_qa.log`. Neither input audio nor the `.djs` archive is changed.

## What is preserved and checked

The archive reader opens only the JSON metadata members. It does not extract or
decode the embedded audio. It rejects unsafe archive member names, duplicate
metadata members, encrypted metadata and metadata exceeding its size limit.
Reading metadata does not verify every embedded audio member's integrity.

The adapter preserves the original cue points, track maps, custom transition
curves, sample/jingle records and unknown native fields as opaque JSON. It
accepts native transition lengths beyond the renderer's 8/16/32-bar choices,
including 24 and 44 bars. Bar counts use explicitly declared matching meters;
four beats per bar are not assumed when meter metadata is absent or inconsistent.

CUE timestamps use **75 frames per second**, including positions with more than
59 minutes. An MP3 named in a `FILE ... WAVE` directive is not assumed to contain
PCM audio. Track titles and count must match the native order before timing is
attached. Multiple-file CUE positions remain file-local and are not combined
into a single export timeline. Unknown CUE directives remain inert text.

## Three different durations

1. `source_selected_span_s` is a span in the source-audio `mixMap`. It is useful
   for checking which original material was selected. Tempo changes mean this
   is **not** its duration in the exported mix. `source_end_beyond_file_s` also
   identifies selections extending beyond the library's source duration.
2. `export_chapter_interval_s` is the interval from one exported CUE marker to
   the next. A previous track can keep playing during the outgoing transition,
   so a chapter interval is **not** its complete audible span.
3. `playback_duration_bounds_s` adds conditional outgoing-overlap bounds to the
   chapter interval. Those bounds use the native project's declared minimum and
   maximum tempos; `local_tempo_playback_duration_bounds_s` uses the narrower
   neighboring-track envelope. These are estimates under stated assumptions,
   not a reconstruction of DJ.Studio's tempo interpolation. The final chapter
   ends at the independently probed/measured export duration, when available.

The CUE markers are assumed to correspond to incoming native entry points;
`cue_marker_entry_alignment_verified` remains false. Unsupported native tempo
modes, effect offsets, overrides or ambiguous effects leave duration unresolved.
Volume, EQ, fades and stems can make part of a selected span effectively
inaudible, even when that span meets the duration threshold.

Minimum-duration results therefore say either **under timing assumptions**, or
**uncertain/unresolved**. `audibility_verified` and `phrase_verified` remain
false. Neither beat numbers nor phrase metadata establish reviewed musical
phrase boundaries. Successful native import and musical listening review are
separate checks.

## Python API

```python
from superdj.djstudio import audit_djstudio, load_djs, read_cue

report = audit_djstudio(
    "project.djs",
    cue="export.cue",
    export_duration_s=3600.0,  # independently measured exported audio
    minimum_track_s=120.0,
    include_native_state=False,
)
```

`load_djs()` returns a `DJStudioProject`. Its `native_state()` method returns an
independent deep copy of all native JSON. `read_cue()` and `parse_cue_text()` parse
chapters without opening any audio paths named inside them. The audit returns a
JSON-serializable dictionary and performs no output writes itself.

## Advanced preserving rewrite

`superdj.project_edit.rewrite_djs(source, new_output, cue_updates=..., transition_updates=...)`
patches explicitly selected zero-based records. Cue fields merge into existing
cueData; duration, automation and lock fields can be updated on a transition.
Unknown metadata and all unmodified members remain. Every retained member is
compared by SHA-256 and the original archive is untouched. Keep personal native
snapshots private.

Archive verification does not confirm application import, exact automation
semantics or audible duration. Import and audition remain separate checks.
Standalone zhilin arrangement uses our mixer for energetic source spans and
long overlaps, without emulating the application's complete rendering engine.
