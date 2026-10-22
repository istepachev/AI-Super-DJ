# Local engines and optional arrangement

## Workstation setup

Use Python 3.12 with FFmpeg/FFprobe and Rubber Band installed separately. Install compatible CPU/CUDA PyTorch builds before optional neural dependencies. `superdj setup --components ../superdj-work/checkouts` obtains pinned audamcp and Bitwize. Keep music, models and outputs outside the source checkout.

The tested neural worker uses torch/torchaudio 2.3.1, torchvision 0.18.1, NumPy 1.26.4, transformers 4.42.3 and timm 1.0.7. Optional constraints also allow compatible PyTorch 2.4/2.5 pairs, which require their own validation. Demucs 4.0.1 is restricted below torch 2.6 because its older checkpoint loader is incompatible with the later default weights-only loading. A separate worker can coexist with a NumPy 2 main environment.

```text
python -m pip install -e ".[grid,repair,cue,stems]"
superdj setup --components ../superdj-work/checkouts --download-model --cache ../superdj-work/cache --download-cue-model
```

`--download-model` enables Beat This! checkpoint downloading. `--download-cue-model` downloads three SHA-256 checked CUE-DETR/processor files into `~/.cache/superdj/cue-detr`. Demucs fetches official `htdemucs` weights on its first enabled inference and subsequently uses the private torch cache. Inference is local; music is not uploaded.

For `make`, create `~/.cache/superdj/runtime.local.json`, replacing all example paths with actual absolute locations:

```json
{
  "audamcp": "/private/checkouts/audamcp",
  "bitwize": "/private/checkouts/bitwize",
  "ffmpeg": "/tools/ffmpeg",
  "ffprobe": "/tools/ffprobe",
  "rubberband": "/tools/rubberband",
  "cache": "/private/cache",
  "output_root": "/private/output"
}
```

On Windows use forward slashes or escaped backslashes in JSON. Optional `cue_model`, `cue_python`, `cue_device`, `stem_python`, `stem_device`, `beat_checkpoint`, `beat_device` choose private models/workers. `SUPERDJ_CUE_MODEL`, `SUPERDJ_CUE_PYTHON`, `SUPERDJ_STEM_PYTHON` override worker fields. CUE-DETR enables when its audited checkpoint is present; Demucs when its worker is configured or the package is installed. Missing engines are disclosed, not listed as executed.

## Structural cues

All source containers use the existing FFmpeg input decoder, including the separate CUE-DETR worker. `superdj.audio` resolves the caller's explicit executable, otherwise `SUPERDJ_FFMPEG`, private runtime `ffmpeg`, then system discovery. Pass the selected executable into the cue job; workers do not need FFmpeg on PATH or extra Python/system codec packages. Decode to mono float PCM at 22.05 kHz for model input and retain the original source. Decoder identity is included in cue configuration/cache keys and provenance so previous librosa-decoded neural cues are not silently reused. See [audio-formats.md](audio-formats.md).

`superdj.cues` adapts upstream CUE-DETR spectrogram/image inference: 22.05 kHz, 128 mel bins, 2048 FFT, 512 hop, 355-frame overlapping patches. GPU batches of 32 bound memory and can introduce small numerical differences from upstream's whole-track batch. Provenance records preprocessing, actual versions, device, hashes, raw detections and normalized relative scores. Relative scores are not probabilities.

Original fusion considers RMS, bass and high-band changes around bar-grid candidates and supporting model cues near those changes. Reviewed imported cues remain exact. It does not snap every detection to a presumed sixteen-bar phrase or label model timing human verified. Beat/downbeat confidence, structural support and phrase review remain separate. CUE-DETR does not determine kick phase or recognize screenshots: it analyzes a spectrogram generated from original audio.

## Voice and frequencies

Automatic windows also use source 35–180 Hz RMS for each detected beat, from a complete mono 8 kHz decode. A seventieth-percentile reference minus 16 dB marks relative bass availability. Require at least 80% supported duration in the relevant parts of each window and no weak run longer than 0.8 seconds near the bass handoff. These heuristic guards permit short fills and avoid taking bass-empty intro/outro regions. They do not classify kicks or certify alignment; bassless sources and explicit quiet locks retain separate judgment. Cache keys include source path/stat and the beat grid.

Demucs estimates vocals using joint mixture standardization, no per-stem peak normalization, zero shifts, 7.8-second segments and 25% overlap. One-second energy profiles inform candidate/section scoring; these are leakage-prone estimates, not calibrated speech probabilities.

Prepared vocals receive the mixture's linear DC/EQ processing and time map. With both prominent estimates available, outgoing voice is suppressed before incoming voice opens; backing remains in the original mixture. Unsupported creative warp or incompatible preparation bypasses subtraction with a recorded reason. Instrumental material does not receive a placebo vocal effect.

Complementary 180/3500 Hz bands reconstruct the input. Paired band gains sum to at most one. Bass transfers over at least eight available shared beats; mids softly hand off over 8–16 beats around 43% of the overlap with 1–2 dB of initial attenuation. Highs cross smoothly throughout the overlap. No whole-deck compensation follows. Adaptive preparation EQ is bounded to 0.8 dB and bypasses neutral controls.

## DJ Жилин and native exceptions

User listening feedback can narrow middle and bass handoffs independently with
per-edge `mid_handoff_bars` and `bass_handoff_beats`. Aim for long useful
harmonic/rhythmic support; shorten for foreground voices or sustained tonal
competition. Pitch-class uncertainty alone is not a reason to remove a groove.
See [adaptive-mids.md](adaptive-mids.md) for the reviewed decision policy and
its current validation limits. This is not a trained taste classifier.

New sets target 210 seconds between actual CUE starts, normally 180–240, for at least 80% of tracks and preferably 90% or more. The outgoing overlap continues into the next chapter; it must not be subtracted from this target. Check the set-wide fraction again on actual rendered CUE sample frames. Ordinary full-deck playback, including both overlaps, has a separate 300-second cap. Named brief and highlighted tracks share the exception budget; a selected highlight may use a 270–300 second chapter with its outgoing overlap added only to that deck’s cap. Required physically short source material is explicitly reported, and does not justify cropping other long sources. See [transition-preferences.md](transition-preferences.md) for the factory policy. Solo time is a distinct remaining body measurement. Automatic windows select an energetic central core using cue confidence, energy and competing vocals/tonal proxy. Shared SYNC overlaps prefer 32 bars, 16 above 20% temporary tempo change, including `--style zhilin`. DJ Zhilin uses broad complementary mid/high curves and a slow bass transfer over one eighth of the overlap, retaining at least sixteen available shared beats in shorter overlaps. Its default packaged profile derives effective EQ-plus-volume shares from the approved reference curves; an explicit private profile can override it. The portable [reference envelopes](../assets/zhilin-reference-v1.json) retain those normalized curves without the original music, source-specific cues or listening feedback; see [friend-setup.md](friend-setup.md) for optional configuration. The same reference is adapted to new music, not claimed as an identical montage. Reduce automatic lengths when a short source or grid would otherwise erase the core. Explicit locks/protected ranges remain constraints.

Reports separate source, rendered deck, solo and overlap durations. The full-play floor is sixty seconds. For playback below two minutes, optional stable phrase loops can add at most sixty seconds/four copies, with identical mixture/voice/drum edits and independent bass/melody join curves. If no candidate passes native harmony/bass/level and actual energy checks, retain the native short play at or above one minute. Locks/protected ranges remain intact. Save checksummed native join context so revision restores the original PCM exactly. Older ten-second/four-bar timeline recipes remain readable; they do not impose a two-minute factory minimum. The legacy preserving `sections.arrange` API remains available for approved source-span montages.

An edge can set `overlap_bars` to 2/4/8/16/32/40/44. New automatic plans choose shared SYNC. Advanced `overlap_beats` integers from 8 to 256 support partial-bar overlaps such as 51 beats; both exact source beat-grid locks are required, mutually exclusive with a bar length. Locks never move. The simple skill surface exposes only style and order preservation.
