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

`superdj.cues` adapts upstream CUE-DETR spectrogram/image inference: 22.05 kHz, 128 mel bins, 2048 FFT, 512 hop, 355-frame overlapping patches. GPU batches of 32 bound memory and can introduce small numerical differences from upstream's whole-track batch. Provenance records preprocessing, actual versions, device, hashes, raw detections and normalized relative scores. Relative scores are not probabilities.

Original fusion considers RMS, bass and high-band changes around bar-grid candidates and supporting model cues near those changes. Reviewed imported cues remain exact. It does not snap every detection to a presumed sixteen-bar phrase or label model timing human verified. Beat/downbeat confidence, structural support and phrase review remain separate. CUE-DETR does not determine kick phase or recognize screenshots: it analyzes a spectrogram generated from original audio.

## Voice and frequencies

Demucs estimates vocals using joint mixture standardization, no per-stem peak normalization, zero shifts, 7.8-second segments and 25% overlap. One-second energy profiles inform candidate/section scoring; these are leakage-prone estimates, not calibrated speech probabilities.

Prepared vocals receive the mixture's linear DC/EQ processing and time map. With both prominent estimates available, outgoing voice is suppressed before incoming voice opens; backing remains in the original mixture. Unsupported creative warp or incompatible preparation bypasses subtraction with a recorded reason. Instrumental material does not receive a placebo vocal effect.

Complementary 180/3500 Hz bands reconstruct the input. Bass uses a complementary swap; mids/highs use separate curves. Incoming mids start quieter and open gradually, with caution on long contrasting transitions. No whole-deck gain compensates the intended soft entry. Adaptive preparation EQ is bounded to 0.8 dB and bypasses neutral controls; absent high frequencies are not synthesized.

## DJ Жилин and native exceptions

`--style zhilin` targets energetic structural source spans of 210 seconds, normally 180–240, preserving approved cue locks and protected ranges. Scoring combines entry confidence, end novelty, energy and competing vocals (or a disclosed tonal proxy). It prefers forty-bar overlaps, reducing to sixteen for distant harmony or insufficient material. Explicit durations override this preference. This is an explainable heuristic, not a trained preference model.

Reports separate source and rendered spans. Tempo/gain can change foreground duration; neither a source span nor a CUE chapter guarantees three minutes of audible solo music. Approved short/long sections remain exceptions with warnings.

An edge can set `overlap_bars` to 2/4/8/16/32/40/44. This explicitly selects F in auto policy. Zhilin's internal `section_overlap_bars` preserves ordinary-first selection. Advanced `overlap_beats` integers from 8 to 256 support partial-bar overlaps such as 51 beats; both source-boundary locks on the supplied beat grid are required and bar length is mutually exclusive. Locks never move. A 51-beat exception is not a complete 4/4 phrase. The simple skill surface exposes only style and order preservation.
