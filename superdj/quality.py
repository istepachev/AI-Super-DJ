"""Independent full-file decode, EBU loudness and oversampled peak checks."""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import subprocess


def execute(args: list[str], log: Path | None = None) -> str:
    result = subprocess.run([str(a) for a in args], capture_output=True)
    output = result.stderr.decode("utf-8", errors="replace")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Audio backend failed ({result.returncode}): {output[-3000:]}")
    return output


def parse_ebu(text: str) -> dict:
    summary = text.rsplit("Summary:", 1)[-1]
    def get(pattern):
        match = re.search(pattern, summary, re.S)
        if not match:
            raise ValueError("Missing full-file EBU summary")
        value = float(match[1])
        return value if math.isfinite(value) else None
    return {"integrated_lufs": get(r"Integrated loudness:.*?I:\s*([-+\w.]+)\s*LUFS"),
            "lra_lu": get(r"Loudness range:.*?LRA:\s*([-+\w.]+)\s*LU"),
            "true_peak_dbfs": get(r"True peak:.*?Peak:\s*([-+\w.]+)\s*dBFS")}


def measure_audio(path: str | Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", log: Path | None = None) -> dict:
    probe = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, check=True)
    data = json.loads(probe.stdout)
    stream = data["streams"][0]
    output = execute([ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-xerror", "-i", str(path),
                      "-filter_complex", "[0:a:0]aformat=sample_fmts=dbl,asplit=2[a][b];[a]ebur128=peak=true:framelog=verbose[e];[b]astats=metadata=0:reset=0[s]",
                      "-map", "[e]", "-map", "[s]", "-f", "null", "-"], log)
    values = parse_ebu(output)
    peaks = re.findall(r"Peak level dB:\s*([-+\w.]+)", output)
    samples = re.findall(r"Number of samples:\s*(\d+)", output)
    return {"path": str(Path(path).resolve()), "codec": stream["codec_name"], "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]), "bits_per_raw_sample": int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0),
            "duration_seconds": float(stream.get("duration") or data["format"]["duration"]),
            "decoded_samples": int(samples[-1]) if samples else None, "sample_peak_dbfs": float(peaks[-1]) if peaks else None,
            "full_decode_ok": True, "measurement": "FFmpeg ebur128 full recording, oversampled true peak", **values}


def export_audio(master: str | Path, output_dir: str | Path, name: str, *, ffmpeg="ffmpeg", ffprobe="ffprobe", maximum_true_peak=-1.0, mp3=True, overwrite=False) -> dict:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    extensions = ["flac", "mp3"] if mp3 else ["flac"]
    if not overwrite and any((directory / f"{name}.{ext}").exists() for ext in extensions):
        raise FileExistsError("Output already exists; choose a new name or explicitly enable overwrite")
    results = {}
    for extension in extensions:
        target = directory / f"{name}.{extension}"
        staged = directory / f".{name}.partial.{extension}"
        gain = 0.0
        for attempt in range(3):
            command = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(master), "-map", "0:a:0", "-map_metadata", "-1"]
            if gain:
                command += ["-af", f"volume={gain:.4f}dB"]
            if extension == "flac":
                command += ["-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-compression_level", "8"]
            else:
                command += ["-c:a", "libmp3lame", "-b:a", "320k"]
            execute(command + [str(staged)])
            measured = measure_audio(staged, ffmpeg=ffmpeg, ffprobe=ffprobe, log=directory / f"{name}_{extension}_qa.log")
            if measured["true_peak_dbfs"] is None:
                raise ValueError("Export produced silent or invalid audio")
            if measured["true_peak_dbfs"] <= maximum_true_peak:
                break
            gain -= measured["true_peak_dbfs"] - maximum_true_peak + 0.15
        else:
            raise RuntimeError("Encoded output still exceeds the true-peak ceiling")
        if extension == "flac" and measured["bits_per_raw_sample"] != 24:
            raise RuntimeError("FLAC did not retain the requested 24-bit depth")
        if extension == 'flac':
            from .integrity import verify_exported_pcm
            measured['master_pcm_verification'] = verify_exported_pcm(master, staged, gain_db=gain)
        staged.replace(target)
        measured["path"] = str(target.resolve())
        measured["export_gain_db"] = gain
        results[extension] = measured
    return results
