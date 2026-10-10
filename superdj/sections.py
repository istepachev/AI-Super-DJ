"""Optional DJ Zhilin section planning, independent of project formats."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np


def arrange(plan: dict, records: list[dict], *, target_seconds=210.0) -> dict:
    """Select 3–4 minute source spans and long, compatible transition regions.

    Imported locks and protected ranges outrank duration preferences. Actual
    output spans are measured by the renderer; source spans are not labelled
    guaranteed foreground/audible durations. Tonality is a vocal-risk proxy.
    """
    if not 180 <= target_seconds <= 240:
        raise ValueError("Zhilin target must be 180–240 seconds")
    result = deepcopy(plan)
    by_id = {str(row["id"]): row for row in records}
    sections, warnings = {}, []
    for position, identity in enumerate(result["order"]):
        row = by_id[identity]
        duration = float(row.get("duration", row.get("duration_seconds")))
        bars = np.asarray(row.get("downbeats", []), dtype=float)
        bpm = float(row.get("bpm") or 0)
        if len(bars) < 10 or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"{identity}: Zhilin needs a usable beat/bar grid")
        cues = row.get("cue_candidates", [])
        structure = row.get("structure", [])
        previous = result["edges"][position - 1] if position else {}
        following = result["edges"][position] if position < len(result["edges"]) else {}
        locked_start = (previous.get("cue_locks") or {}).get("incoming_start_seconds")
        locked_end = (following.get("cue_locks") or {}).get("outgoing_end_seconds")
        protected = row.get("protected_ranges", [])
        starts = [float(locked_start)] if locked_start is not None else sorted(set(
            [float(bars[0])] + [float(c["time"]) for c in cues if float(c["time"]) < max(1, duration - 180)]))
        choices = []
        for start in starts:
            ends = [float(locked_end)] if locked_end is not None else bars[(bars > start) & (bars - start <= 240 + 1e-6)].tolist()
            if not ends and locked_end is None:
                ends = [float(bars[-1])]
            for end in ends:
                if not 0 <= start < end <= duration:
                    continue
                if locked_start is None and locked_end is None and end - start < min(180, duration - bars[0] - 2):
                    continue
                if any(start > float(r[0]) or end < float(r[1]) for r in protected):
                    continue
                boundary = [c for c in cues if abs(float(c["time"]) - end) < .05]
                novelty = max((float(c.get("novelty") or 0) for c in boundary), default=0)
                entry = max((float(c.get("confidence") or 0) for c in cues if abs(float(c["time"]) - start) < .05), default=0)
                # Select energetic sections, rather than simply skipping the intro.
                content = [s for s in structure if start <= float(s["time"]) < end]
                energy = float(np.mean([s["energy"] for s in content])) if content else 0
                transition_beats = 160
                transition_seconds = transition_beats * 60 / bpm
                boundary_content = [s for s in content if float(s["time"]) < start + transition_seconds or float(s["time"]) > end - transition_seconds]
                activity = float(np.mean([s.get("vocal_activity", s.get("tonal_activity_proxy", 0)) for s in boundary_content])) if boundary_content else 0
                score = abs(end - start - target_seconds) / 30 - .6 * novelty - .6 * entry - .4 * energy + .8 * activity
                choices.append((score, start, end))
        if not choices:
            raise ValueError(f"{identity}: duration/cue/protected-range constraints leave no section")
        _, start, end = min(choices)
        sections[identity] = {"start": start, "end": end, "source_span_seconds": end - start,
                              "target_seconds": target_seconds, "protected_ranges": protected,
                              "source_duration_in_range": 180 <= end - start <= 240,
                              "locked_start": locked_start is not None, "locked_end": locked_end is not None}
        if not 180 <= end - start <= 240:
            warnings.append(f"{identity}: source/approved locks require a duration exception")
    for edge in result["edges"]:
        a, b = sections[edge["from"]], sections[edge["to"]]
        edge.setdefault("cue_locks", {}).update({"outgoing_end_seconds": a["end"], "incoming_start_seconds": b["start"]})
        # 40 bars is a style preference. A short source or distant harmony can
        # reduce it; actual grid validity is still checked at render time.
        if "overlap_bars" not in edge and "overlap_beats" not in edge:
            maximum = min(a["source_span_seconds"] * by_id[edge["from"]]["bpm"] / 60,
                          b["source_span_seconds"] * by_id[edge["to"]]["bpm"] / 60)
            count = 16 if "distant" in str(edge.get("harmonic_relation", "")) else 40
            if maximum <= 2 * count * 4 + 8:
                count = 16
            edge["section_overlap_bars"] = count
    result.update({"style": "zhilin", "sections": sections,
                   "section_warnings": warnings,
                   "section_policy": "3–4 minute source spans; structural cues, low competing tonal activity, long overlaps; approved exceptions preserved"})
    return result
