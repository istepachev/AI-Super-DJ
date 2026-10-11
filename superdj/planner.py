"""Transparent whole-playlist planning from measured musical features.

This module has no audio dependencies. Its scores are engineering heuristics,
not probabilities of a good transition. For small playlists Held--Karp finds
the exact minimum of the declared objective, including the position energy
term. Larger playlists use a bounded beam search and segment improvement.
Neither method claims to infer phrase boundaries or to reproduce proprietary
software. A rendering engine must verify beat/phrase grids separately.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import random
import re
from typing import Any, Mapping, Sequence


DEFAULT_WEIGHTS = {
    "harmonic": 2.0,
    "tempo": 1.0,
    "energy_step": 0.25,
    "energy_arc": 0.60,
    "tempo_arc": 1.0,
    "bass_arc": 0.60,
    "similarity": 0.35,
    "uncertainty": 0.60,
    "vocal_overlap": 0.80,
    "tempo_reset": 8.0,
    "creative_exit": 2.0,
}
_KEYS = {
    "B major": "1B", "F# major": "2B", "Gb major": "2B",
    "Db major": "3B", "C# major": "3B", "Ab major": "4B",
    "G# major": "4B", "Eb major": "5B", "D# major": "5B",
    "Bb major": "6B", "A# major": "6B", "F major": "7B",
    "C major": "8B", "G major": "9B", "D major": "10B",
    "A major": "11B", "E major": "12B", "Ab minor": "1A",
    "G# minor": "1A", "Eb minor": "2A", "D# minor": "2A",
    "Bb minor": "3A", "A# minor": "3A", "F minor": "4A",
    "C minor": "5A", "G minor": "6A", "D minor": "7A",
    "A minor": "8A", "E minor": "9A", "B minor": "10A",
    "F# minor": "11A", "Gb minor": "11A", "C# minor": "12A",
    "Db minor": "12A",
}


@dataclass(frozen=True)
class _Track:
    id: str
    path: str | None
    bpm: float | None
    key: tuple[int, str] | None
    energy: float
    energy_observed: bool
    energy_method: str
    bass_intensity: float | None
    bass_method: str
    key_confidence: float
    bpm_confidence: float
    embedding: tuple[float, ...] | None
    boundary: Mapping[str, Any]
    notices: tuple[str, ...]


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _unit(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _normalise(record: Mapping[str, Any]) -> _Track:
    identity = str(record.get("id") or record.get("path") or "")
    if not identity:
        raise ValueError("Every track needs a nonempty id or path")
    notices: list[str] = []
    bpm = record.get("bpm")
    if bpm is not None:
        bpm = _finite(bpm, f"{identity}.bpm")
        if bpm <= 0:
            raise ValueError(f"{identity}.bpm must be positive")
    else:
        notices.append("BPM unknown; this edge cannot promise beatmatching")
    raw_key = str(record.get("key_camelot") or record.get("key") or "").strip()
    raw_key = _KEYS.get(raw_key, raw_key).upper()
    match = re.fullmatch(r"(1[0-2]|[1-9])([AB])", raw_key)
    key = (int(match[1]), match[2]) if match else None
    if key is None:
        notices.append("Key unknown or unsupported")
    if record.get("key_confidence") is None:
        notices.append("Key confidence missing; conservative heuristic 0.5 used")
    if record.get("bpm_confidence") is None:
        notices.append("BPM confidence missing; conservative heuristic 0.5 used")
    key_conf = _unit(record.get("key_confidence", 0.5) or 0, "key_confidence")
    bpm_conf = _unit(record.get("bpm_confidence", 0.5) or 0, "bpm_confidence")
    if record.get("key_confidence") is None:
        key_conf = 0.5
    if record.get("bpm_confidence") is None:
        bpm_conf = 0.5
    if key is None:
        key_conf = 0.0
    if bpm is None:
        bpm_conf = 0.0
    intensity = record.get("intensity", record.get("energy"))
    energy_method = str(record.get("energy_method") or
                        ("caller-supplied intensity/energy" if intensity is not None else "unavailable"))
    energy_observed = intensity is not None
    if intensity is not None:
        energy = _unit(intensity, f"{identity}.intensity")
    else:
        energy = 0.5
        notices.append("Intensity evidence missing; no intensity preference applied")
    if record.get("intensity") is None and "RMS level proxy" in energy_method:
        energy_observed = False
        notices.append("Legacy RMS level is not musical intensity; no intensity preference applied")
    bass = record.get("bass_intensity")
    if bass is not None:
        bass = _unit(bass, f"{identity}.bass_intensity")
    else:
        notices.append("Bass evidence missing; no bass preference applied")
    embedding = record.get("embedding", record.get("emb"))
    if isinstance(embedding, Mapping):
        embedding = embedding.get("values", embedding.get("vector"))
    if embedding is not None:
        embedding = tuple(_finite(v, "embedding") for v in embedding)
        if not embedding or not sum(v * v for v in embedding):
            embedding = None
    boundary = record.get("boundary_features") or {}
    if not isinstance(boundary, Mapping):
        raise ValueError("boundary_features must be an object")
    return _Track(
        identity, str(record["path"]) if record.get("path") else None,
        bpm, key, energy, energy_observed, energy_method, bass,
        str(record.get("bass_intensity_method") or
            ("caller-supplied bass_intensity" if bass is not None else "unavailable")),
        key_conf, bpm_conf, embedding, boundary, tuple(notices),
    )


def _weights(overrides: Mapping[str, float] | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    for name, value in (overrides or {}).items():
        if name not in weights:
            raise ValueError(f"Unknown objective weight: {name}")
        number = _finite(value, name)
        if number < 0:
            raise ValueError(f"{name} must be nonnegative")
        weights[name] = number
    return weights


def _harmonic(a: _Track, b: _Track) -> tuple[float, str]:
    if a.key is None or b.key is None:
        return 0.75, "unknown key relationship"
    number_a, letter_a = a.key
    number_b, letter_b = b.key
    distance = min(abs(number_a - number_b), 12 - abs(number_a - number_b))
    if a.key == b.key:
        return 0.0, "same estimated key"
    if distance == 0:
        return 0.08, "relative major/minor"
    if distance == 1 and letter_a == letter_b:
        return 0.12, "adjacent Camelot key"
    return min(1.0, 0.35 + distance * 0.13 + (letter_a != letter_b) * 0.16), "distant key; melodic overlap needs review"


def _tempo(a: _Track, b: _Track, limit: float,
           ramp_limit: float | None = None) -> dict[str, Any]:
    if a.bpm is None or b.bpm is None:
        return {"beatmatch_possible": False, "target_bpm": None,
                "outgoing_rate": 1.0, "incoming_rate": 1.0,
                "max_change": 0.0, "reason": "unknown BPM"}
    # Intersection is the actual feasibility test. Geometric midpoint alone
    # can reject feasible edges because positive/negative rate limits differ.
    lower = max(a.bpm, b.bpm) * (1 - limit)
    upper = min(a.bpm, b.bpm) * (1 + limit)
    if lower > upper + 1e-9:
        if ramp_limit is not None:
            # The harmonic mean minimizes the largest relative deck change.
            target = 2 * a.bpm * b.bpm / (a.bpm + b.bpm)
            outgoing, incoming = target / a.bpm, target / b.bpm
            change = max(abs(outgoing - 1), abs(incoming - 1))
            if change <= ramp_limit + 1e-9:
                return {"beatmatch_possible": True, "target_bpm": target,
                        "outgoing_rate": outgoing, "incoming_rate": incoming,
                        "max_change": change, "technique": "tempo_ramp",
                        "applied_limit": ramp_limit,
                        "reason": "common tempo with a bounded smooth tempo ramp"}
        if 0.65 <= b.bpm / a.bpm < 1:
            return {"beatmatch_possible": False, "target_bpm": b.bpm,
                    "outgoing_rate": b.bpm / a.bpm, "incoming_rate": 1.0,
                    "max_change": 1 - b.bpm / a.bpm,
                    "technique": "outgoing_only_vinyl",
                    "reason": "preserve incoming native tempo; outgoing exit needs a short creative speed effect"}
        return {"beatmatch_possible": False, "target_bpm": None,
                "outgoing_rate": 1.0, "incoming_rate": 1.0,
                "max_change": 0.0, "reason": "no common tempo inside stretch limit"}
    target = min(upper, max(lower, math.sqrt(a.bpm * b.bpm)))
    outgoing, incoming = target / a.bpm, target / b.bpm
    return {"beatmatch_possible": True, "target_bpm": target,
            "outgoing_rate": outgoing, "incoming_rate": incoming,
            "max_change": max(abs(outgoing - 1), abs(incoming - 1)),
            "technique": "standard_blend", "applied_limit": limit,
            "reason": "common tempo inside stretch limit"}


def _edge(a: _Track, b: _Track, limit: float, weights: Mapping[str, float],
          allow_reset: bool, ramp_limit: float | None = None) -> dict[str, Any]:
    tempo = _tempo(a, b, limit, ramp_limit)
    harmonic_base, harmonic_reason = _harmonic(a, b)
    confidence = min(a.key_confidence, b.key_confidence)
    harmonic = confidence * harmonic_base + (1 - confidence) * 0.75
    bpm_confidence = min(a.bpm_confidence, b.bpm_confidence)
    uncertainty = (2 - confidence - bpm_confidence) / 2
    energy_step = abs(a.energy - b.energy) if a.energy_observed and b.energy_observed else 0.0
    similarity = 0.0
    has_similarity = bool(a.embedding and b.embedding and len(a.embedding) == len(b.embedding))
    if has_similarity:
        dot = sum(x * y for x, y in zip(a.embedding, b.embedding))
        norm = math.sqrt(sum(x * x for x in a.embedding) * sum(y * y for y in b.embedding))
        similarity = (1 - max(-1.0, min(1.0, dot / norm))) / 2
    outgoing_vocal = _unit(a.boundary.get("outro_vocal", 0), "outro_vocal")
    incoming_vocal = _unit(b.boundary.get("intro_vocal", 0), "intro_vocal")
    vocal_overlap = min(outgoing_vocal, incoming_vocal)
    tempo_term = tempo["max_change"] / max(limit, 0.01)
    reset = not tempo["beatmatch_possible"]
    creative_exit = tempo.get("technique") == "outgoing_only_vinyl"
    components = {
        "harmonic": harmonic * weights["harmonic"],
        "tempo": tempo_term * weights["tempo"],
        "energy_step": energy_step * weights["energy_step"],
        "similarity": similarity * weights["similarity"],
        "uncertainty": uncertainty * weights["uncertainty"],
        "vocal_overlap": vocal_overlap * weights["vocal_overlap"],
        "tempo_reset": weights["tempo_reset"] if reset and not creative_exit else 0.0,
        "creative_exit": weights["creative_exit"] if creative_exit else 0.0,
    }
    reasons = [harmonic_reason, tempo["reason"]]
    if confidence < 0.65:
        reasons.append("low key confidence; harmonic match is uncertain")
    if bpm_confidence < 0.65:
        reasons.append("low BPM confidence; verify the beat grid")
    if vocal_overlap > 0.4:
        reasons.append("two vocal boundaries may overlap")
    if has_similarity:
        reasons.append("acoustic embedding similarity considered")
    if reset:
        reasons.append("review the short outgoing-only effect cue" if creative_exit else "no bounded common tempo; select another continuous transition technique")
    cost = sum(components.values()) if allow_reset or not reset else math.inf
    return {"from": a.id, "to": b.id, "cost": cost, "components": components,
            "tempo": tempo, "tempo_reset": reset, "transition_review_required": reset,
            "harmonic_relation": harmonic_reason,
            "confidence": {"key": confidence, "bpm": bpm_confidence},
            "reasons": reasons}


def transition_score(a: Mapping[str, Any], b: Mapping[str, Any], *,
                     max_tempo_change: float = 0.06,
                     weights: Mapping[str, float] | None = None,
                     allow_tempo_reset: bool = True,
                     tempo_ramp_max_change: float | None = None) -> dict[str, Any]:
    """Explain one edge with distinct ordinary and smooth-ramp tempo caps."""
    limit = _validate_limit(max_tempo_change)
    ramp_limit = _ramp_limit(tempo_ramp_max_change, limit)
    edge = _edge(_normalise(a), _normalise(b), limit, _weights(weights), allow_tempo_reset, ramp_limit)
    return _json_edge(edge)


def _validate_limit(value: float) -> float:
    limit = _finite(value, "max_tempo_change")
    if not 0 <= limit < 0.5:
        raise ValueError("max_tempo_change must be at least 0 and below 0.5")
    return limit


def _ramp_limit(value: float | None, regular: float) -> float | None:
    if value is None:
        return None
    result = _validate_limit(value)
    if result < regular:
        raise ValueError("tempo_ramp_max_change must be at least the regular tempo limit")
    return result


def _arc_fraction(fraction: float, arc: str) -> float:
    if arc == "release":
        return 1 - fraction
    if arc == "wave":
        # An early climb, a breathing valley, then the strongest ending.
        for start, end, low, high in ((0, .42, 0, .82), (.42, .65, .82, .35), (.65, 1, .35, 1)):
            if fraction <= end:
                return low + (high-low) * (fraction-start) / (end-start)
    return fraction


def _targets(tracks: Sequence[_Track], arc: str | Sequence[float],
             feature: str = "energy") -> list[float]:
    count = len(tracks)
    if not isinstance(arc, str):
        if feature != "energy":
            return []
        result = [_unit(v, "energy_arc") for v in arc]
        if len(result) != count:
            raise ValueError("Custom energy_arc needs one target per track")
        return result
    if arc not in {"build", "release", "wave", "none", "flat"}:
        raise ValueError("energy_arc must be build, release, wave, flat, none or a target list")
    if arc == "none":
        return []
    values = sorted(getattr(t, feature) for t in tracks
                    if getattr(t, feature) is not None and (feature != "energy" or t.energy_observed))
    if not values:
        return []
    if arc == "flat":
        return [values[len(values) // 2]] * count
    low, high = values[0], values[-1]
    return [low + (high-low) * _arc_fraction(p / max(1, count-1), arc) for p in range(count)]


def _order_cost(order: Sequence[int], costs: Sequence[Sequence[float]],
                position_costs: Sequence[Sequence[float]]) -> float:
    return sum(position_costs[p][i] for p, i in enumerate(order)) + sum(
        costs[a][b] for a, b in zip(order, order[1:]))


def _greedy(costs: Sequence[Sequence[float]], nodes: Sequence[Sequence[float]],
            first: int | None, last: int | None, ranks: Mapping[int, int]) -> list[int] | None:
    n = len(costs)
    initial = [first] if first is not None else [i for i in range(n) if i != last or n == 1]
    start = min(initial, key=lambda i: (nodes[0][i], ranks[i]))
    order = [start]
    unused = set(range(n)) - {start}
    while unused:
        available = [i for i in unused if i != last or len(unused) == 1]
        choice = min(available, key=lambda i: (costs[order[-1]][i] + nodes[len(order)][i], ranks[i]))
        if not math.isfinite(costs[order[-1]][choice]):
            return None
        order.append(choice)
        unused.remove(choice)
    return order


def _exact(costs: Sequence[Sequence[float]], nodes: Sequence[Sequence[float]],
           first: int | None, last: int | None, ranks: Mapping[int, int]) -> list[int]:
    n = len(costs)
    size = (1 << n) * n
    best = array("d", [math.inf]) * size
    previous = array("h", [-1]) * size
    starts = [first] if first is not None else sorted(range(n), key=ranks.get)
    for i in starts:
        if i != last or n == 1:
            best[(1 << i) * n + i] = nodes[0][i]
    full = (1 << n) - 1
    for mask in range(1, full + 1):
        count = mask.bit_count()
        if count == n:
            continue
        remaining = full ^ mask
        if last is not None and count < n - 1:
            remaining &= ~(1 << last)
        predecessors = mask
        while predecessors:
            low_bit = predecessors & -predecessors
            predecessor = low_bit.bit_length() - 1
            predecessors ^= low_bit
            value = best[mask * n + predecessor]
            if not math.isfinite(value):
                continue
            candidates = remaining
            while candidates:
                next_bit = candidates & -candidates
                candidate = next_bit.bit_length() - 1
                candidates ^= next_bit
                proposal = value + costs[predecessor][candidate] + nodes[count][candidate]
                index = (mask | next_bit) * n + candidate
                old_parent = previous[index]
                if proposal < best[index] - 1e-12 or (
                    math.isfinite(proposal) and abs(proposal - best[index]) <= 1e-12
                    and (old_parent < 0 or ranks[predecessor] < ranks[old_parent])
                ):
                    best[index] = proposal
                    previous[index] = predecessor
    ends = [last] if last is not None else list(range(n))
    end = min(ends, key=lambda i: (best[full * n + i], ranks[i]))
    if not math.isfinite(best[full * n + end]):
        raise ValueError("No complete path meets the locked endpoints and tempo constraint")
    order = []
    mask = full
    while end >= 0:
        order.append(end)
        parent = previous[mask * n + end]
        mask ^= 1 << end
        end = parent
    return list(reversed(order))


def _beam(costs: Sequence[Sequence[float]], nodes: Sequence[Sequence[float]],
          first: int | None, last: int | None, ranks: Mapping[int, int], width: int) -> list[int]:
    n = len(costs)
    starts = [first] if first is not None else [i for i in range(n) if i != last or n == 1]
    states = sorted([(nodes[0][i], 1 << i, (i,)) for i in starts],
                    key=lambda item: (item[0], tuple(ranks[i] for i in item[2])))[:width]
    for position in range(1, n):
        unique: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {}
        for value, mask, path in states:
            for candidate in range(n):
                if mask & (1 << candidate) or candidate == last and position < n - 1:
                    continue
                proposal = value + costs[path[-1]][candidate] + nodes[position][candidate]
                if not math.isfinite(proposal):
                    continue
                new_mask = mask | (1 << candidate)
                key = (new_mask, candidate)
                entry = (proposal, new_mask, path + (candidate,))
                old = unique.get(key)
                if old is None or proposal < old[0]:
                    unique[key] = entry
        states = sorted(unique.values(), key=lambda item: (
            item[0], tuple(ranks[i] for i in item[2])))[:width]
        if not states:
            raise ValueError("Beam search found no complete feasible path; widen the beam or allow tempo resets")
    path = list(states[0][2])
    # Evaluate each full directional path: reversing a segment changes both
    # directed edges and the position energy term, so symmetric-TSP shortcuts
    # would compute the wrong objective here.
    start_bound = 1 if first is not None else 0
    end_bound = n - 1 if last is not None else n
    for _ in range(8):
        old_cost = _order_cost(path, costs, nodes)
        improved_path = path
        improved_cost = old_cost
        for a in range(start_bound, end_bound - 1):
            for b in range(a + 2, end_bound + 1):
                candidate = path[:a] + list(reversed(path[a:b])) + path[b:]
                value = _order_cost(candidate, costs, nodes)
                if value < improved_cost - 1e-12:
                    improved_path, improved_cost = candidate, value
        path = improved_path
        if improved_cost >= old_cost - 1e-12:
            break
    return path


def _json_edge(edge: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(edge)
    if not math.isfinite(result["cost"]):
        result["cost"] = None
        result["feasible"] = False
    else:
        result["cost"] = round(result["cost"], 6)
        result["feasible"] = True
    result["components"] = {name: round(value, 6) for name, value in edge["components"].items()}
    result["tempo"] = {name: round(value, 8) if isinstance(value, float) else value
                       for name, value in edge["tempo"].items()}
    return result


def plan_tracks(records: Sequence[Mapping[str, Any]], *, mode: str = "auto",
                fixed_order: Sequence[str] | None = None, first_id: str | None = None,
                last_id: str | None = None, seed: int = 0, max_tempo_change: float = 0.06,
                energy_arc: str | Sequence[float] = "build", allow_tempo_reset: bool = True,
                weights: Mapping[str, float] | None = None, exact_limit: int = 17,
                beam_width: int = 256,
                tempo_ramp_max_change: float | None = None) -> dict[str, Any]:
    """Plan every input once, or diagnose a user-defined fixed order.

    ``id`` defaults to ``path``. BPM may be unknown; key accepts standard
    Camelot or names such as ``C# minor``. Intensity/energy, bass_intensity
    and confidences must be [0, 1]. Measured intensity takes precedence over
    energy; legacy analysis RMS energy is excluded from musical preferences.
    Confidences are caller-supplied heuristic scores, not calibrated odds.
    Optional embedding and boundary_features.intro_vocal/outro_vocal improve
    pair costs. Auto does not change tracks or discard difficult tracks.

    The +/- tempo caps apply to *each deck at the common transition tempo*.
    Larger changes use a distinct bounded smooth-ramp technique. Unreachable
    pairs become explicit, expensive native-tempo bridge candidates at rate 1, or
    make the path infeasible when allow_tempo_reset=False. Candidates in each
    edge report are ranked by immediate cost, not by a promised alternate
    globally optimal playlist. Named build/wave arcs prefer slower, lighter
    starts and faster, stronger endings within the measured input range;
    wave includes an intervening valley. Missing evidence has zero position
    cost. Targets are preferences, not edits to playback tempo or audio.
    Seed resolves equal-cost choices only.
    """
    if mode not in {"auto", "fixed"}:
        raise ValueError("mode must be auto or fixed")
    if mode == "auto" and fixed_order is not None:
        raise ValueError("fixed_order requires mode='fixed'")
    if not 1 <= exact_limit <= 18:
        raise ValueError("exact_limit must be between 1 and 18")
    if not 1 <= beam_width <= 5000:
        raise ValueError("beam_width must be between 1 and 5000")
    tracks = [_normalise(record) for record in records]
    if not tracks:
        raise ValueError("At least one track is required")
    ids = [t.id for t in tracks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate track ids are not allowed")
    paths = [t.path for t in tracks if t.path]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source paths are not allowed")
    index = {identity: position for position, identity in enumerate(ids)}
    for identity in (first_id, last_id):
        if identity is not None and identity not in index:
            raise ValueError(f"Unknown locked endpoint: {identity}")
    if first_id is not None and first_id == last_id and len(tracks) > 1:
        raise ValueError("Different endpoints are required for a multi-track set")
    first, last = index.get(first_id), index.get(last_id)
    limit = _validate_limit(max_tempo_change)
    ramp_limit = _ramp_limit(tempo_ramp_max_change, limit)
    objective_weights = _weights(weights)
    targets = _targets(tracks, energy_arc)
    tempo_targets = _targets(tracks, energy_arc, "bpm")
    bass_targets = _targets(tracks, energy_arc, "bass_intensity")
    known_bpms = [t.bpm for t in tracks if t.bpm is not None]
    tempo_span = max(known_bpms)-min(known_bpms) if known_bpms else 0.0
    position_parts = [[{
        "intensity": abs(t.energy-targets[p]) * objective_weights["energy_arc"]
                     if targets and t.energy_observed else 0.0,
        "tempo": abs(t.bpm-tempo_targets[p]) / tempo_span * objective_weights["tempo_arc"]
                 if tempo_targets and t.bpm is not None and tempo_span > 0 else 0.0,
        "bass": abs(t.bass_intensity-bass_targets[p]) * objective_weights["bass_arc"]
                if bass_targets and t.bass_intensity is not None else 0.0,
    } for t in tracks] for p in range(len(tracks))]
    nodes = [[sum(parts.values()) for parts in row] for row in position_parts]
    edges = [[_edge(a, b, limit, objective_weights, allow_tempo_reset, ramp_limit)
              for b in tracks] for a in tracks]
    costs = [[edge["cost"] for edge in row] for row in edges]
    tie_order = list(range(len(tracks)))
    random.Random(seed).shuffle(tie_order)
    ranks = {value: rank for rank, value in enumerate(tie_order)}
    baseline = _greedy(costs, nodes, first, last, ranks)
    if mode == "fixed":
        fixed_ids = list(fixed_order) if fixed_order is not None else ids
        if len(fixed_ids) != len(ids) or set(fixed_ids) != set(ids):
            raise ValueError("fixed_order must contain every input id exactly once")
        order = [index[identity] for identity in fixed_ids]
        if first is not None and order[0] != first or last is not None and order[-1] != last:
            raise ValueError("Fixed order conflicts with a locked endpoint")
        if not math.isfinite(_order_cost(order, costs, nodes)):
            raise ValueError("Fixed order violates the strict tempo constraint")
        method = "fixed order, preserved exactly"
        optimal = None
    elif len(tracks) <= exact_limit:
        order = _exact(costs, nodes, first, last, ranks)
        method = "exact Held-Karp with position intensity, tempo and bass costs"
        optimal = True
    else:
        try:
            order = _beam(costs, nodes, first, last, ranks, beam_width)
        except ValueError:
            if baseline is None:
                raise
            order = baseline
        # A bounded search must at least keep the same-objective baseline if
        # it is feasible; this is an engineering guarantee, not a music claim.
        if baseline is not None and _order_cost(baseline, costs, nodes) < _order_cost(order, costs, nodes):
            order = baseline
        method = f"beam search width {beam_width} with directional segment improvement"
        optimal = False
    total = _order_cost(order, costs, nodes)
    baseline_cost = _order_cost(baseline, costs, nodes) if baseline is not None else None
    reports = []
    used = {order[0]}
    for position, (a, b) in enumerate(zip(order, order[1:]), start=1):
        report = _json_edge(edges[a][b])
        report["position"] = position
        report["position_energy_cost"] = round(position_parts[position][b]["intensity"], 6)
        report["position_arc_components"] = {k:round(v, 6) for k,v in position_parts[position][b].items()}
        report["position_arc_cost"] = round(nodes[position][b], 6)
        report["selected_step_cost"] = round(costs[a][b] + nodes[position][b], 6)
        available = [i for i in range(len(tracks)) if i not in used
                     and (i != last or position == len(tracks) - 1)
                     and math.isfinite(costs[a][i])]
        ranked = sorted(available, key=lambda i: (costs[a][i] + nodes[position][i], ranks[i]))
        report["selected_immediate_rank"] = ranked.index(b) + 1
        report["candidates_scope"] = "remaining tracks, immediate edge plus next-position intensity/tempo/bass cost; not future-route evaluation"
        report["top7_candidates"] = [
            {"id": tracks[i].id, "step_cost": round(costs[a][i] + nodes[position][i], 6),
             "selected": i == b, "reasons": edges[a][i]["reasons"]}
            for i in ranked[:7]
        ]
        # Compatibility for existing readers; the principal shortlist is seven.
        report["top3_candidates"] = report["top7_candidates"][:3]
        reports.append(report)
        used.add(b)
    reset_count = sum(report["tempo_reset"] for report in reports)
    low_confidence = [t.id for t in tracks if min(t.key_confidence, t.bpm_confidence) < 0.65]
    warnings = [f"{t.id}: {notice}" for t in tracks for notice in t.notices]
    if low_confidence:
        warnings.append("Low-confidence musical analysis needs review; confidence is not a calibrated probability")
    if reset_count:
        warnings.append(f"{reset_count} transitions use or need a separately reviewed creative technique; they are not ordinary 1:1 common-tempo blends")
    warnings.append("This plan does not validate bar-one or phrase boundaries; verify them before rendering")
    return {
        "schema_version": 1, "mode": mode, "method": method,
        "optimal_for_declared_objective": optimal,
        "order": [tracks[i].id for i in order],
        "ordered_paths": [tracks[i].path for i in order],
        "total_cost": round(total, 6),
        "baseline_order": [tracks[i].id for i in baseline] if baseline is not None else None,
        "baseline_cost": round(baseline_cost, 6) if baseline_cost is not None else None,
        "improvement": round(baseline_cost - total, 6) if baseline_cost is not None else None,
        "baseline_description": "greedy with identical objective, stretch limit, seed and endpoint locks; cost is not audible quality",
        "weights": objective_weights, "energy_targets": targets,
        "tempo_targets_bpm": tempo_targets, "bass_targets": bass_targets,
        "set_arc": {"shape": energy_arc if isinstance(energy_arc, str) else "custom intensity only",
                    "scope": "position preferences in route search; native tempo and audio are unchanged",
                    "tempo_normalisation": "absolute deviation divided by known input BPM span",
                    "missing_evidence": "zero contribution; no guessed bass or intensity",
                    "positions": [{"id":tracks[i].id, "intensity":tracks[i].energy if tracks[i].energy_observed else None,
                                   "intensity_method":tracks[i].energy_method,
                                   "bpm":tracks[i].bpm, "bass_intensity":tracks[i].bass_intensity,
                                   "bass_method":tracks[i].bass_method,
                                   "target_intensity":targets[p] if targets else None,
                                   "target_bpm":tempo_targets[p] if tempo_targets else None,
                                   "target_bass":bass_targets[p] if bass_targets else None,
                                   "components":{k:round(v, 6) for k,v in position_parts[p][i].items()}}
                                  for p,i in enumerate(order)]}, "edges": reports,
        "uncertainty": {"low_confidence_tracks": low_confidence, "warnings": warnings},
        "constraints": {"first_id": first_id, "last_id": last_id,
                        "max_tempo_change_per_deck": limit,
                        "tempo_ramp_max_change_per_deck": ramp_limit,
                        "allow_tempo_reset": allow_tempo_reset,
                        "all_tracks_once": True, "seed": seed},
        "tempo_reset_count": reset_count,
        "tempo_ramp_count": sum(report["tempo"].get("technique") == "tempo_ramp" for report in reports),
        "render_readiness": "needs_transition_review" if reset_count else "candidate_pending_grid_and_phrase_review",
    }
