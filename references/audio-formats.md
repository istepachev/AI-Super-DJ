# Audio input contract

The prepared workstation uses one existing FFmpeg build for codec decoding.
No per-format Python package, OS codec pack, PATH edit, or conversion of the
original library is required. Initial setup of the Python engine, FFmpeg and
optional neural models remains a separate one-time workstation preparation;
the skill's Markdown instructions do not contain those binaries or weights.

## Discovery and actual decoding

`superdj.audio.AUDIO_EXTENSIONS` is the shared list for folder selection and
completed-set identity normalization. It includes FLAC; MP3/MP2/MP1;
WAV/WAVE/BWF/RF64/W64; AIF/AIFF/AIFC; M4A/M4B/AAC; OGG/OGA/OPUS;
WMA/APE/WV/TTA; CAF/AU/SND/AMR/AC3/EAC3; MP4/MKA/WebM; DSF/DFF.
Support for a particular codec is determined by the configured FFmpeg build.
Containers such as MP4 or WebM must contain an audio stream.

FFmpeg inspects the actual contents, not an inferred codec from the suffix.
AAC and ALAC in M4A use the same input path. A recording renamed to
`track.m4a.mp3` is decoded from its actual container. An arbitrary unknown
extension is not automatically included during folder selection.

`ffmpeg_executable` retains an explicit caller-supplied executable. Otherwise
it resolves `SUPERDJ_FFMPEG`, then `runtime.local.json`'s `ffmpeg`, then the
system executable. A missing explicit/configured executable is an error,
not permission to silently switch engines or install another codec stack.

## Main analysis and separate workers

Full-track analysis decodes the first audio stream to a private mono raw
float PCM file at 22.05 kHz and memory maps it. CUE-DETR decodes the same
first stream to mono float PCM at 22.05 kHz using the explicitly passed
executable. Librosa operates on this decoded array for spectral features;
it does not open source containers. Preparation and Demucs already decode
source containers with the same selected FFmpeg before reading PCM WAV.

Cue-job configuration/cache keys include the decoder identity. Cue provenance
records its executable, stream, sample rate and decoded sample count. The
original remains unchanged. Analysis resampling is distinct from the final
render/export rate and does not restore detail lost in a lossy recording.

Missing audio, invalid containers, decoder errors and empty PCM stop that
input with a file-specific error. No silence is substituted. Nonfinite
model input is rejected. Raw headerless PCM requires its source parameters;
encrypted or unsupported media is not automatically made readable.

## Verification

`tests/test_audio_formats.py` encodes and reads actual generated signals in
19 container/codec combinations, including both AAC and ALAC in M4A, with
PATH empty and no per-format decoding package. It checks usable samples,
duration, spectral content, unchanged input bytes, mislabeled M4A as MP3,
equivalent pipe/file PCM, clean failure and original/output preservation.
Additional history tests preserve repeat filtering across the new suffixes.

For a configured neural workstation, also run the real CUE-DETR worker on
M4A without FFmpeg on PATH. Keep the source hash, decoder provenance and
returned cue times in private validation output. Synthetic codec checks
do not certify the musical quality of a completed mix.
