"""Auditable BPM/key sequencing inspired by published Harmonize behavior.

Relation costs, weights and search are ours, not DJ.Studio internals. Fuzzy
and Mood are explicit experimental profiles; official descriptions conflict.
Audio, energy, confidence and phrase data do not enter this order objective.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Mapping, Sequence

from .planner import _KEYS, _beam, _exact, _greedy, _order_cost


VERSION = "superdj-harmonize-3"
RELATION_COSTS = {
    "same_key": 0.0, "relative_major_minor": 0.10,
    "perfect_boost": 0.15, "perfect_drop": 0.15,
    "diagonal": 0.30, "energy_boost": 0.45,
    "parallel_major_minor": 0.55, "semitone_lift": 0.90,
    "incompatible": 1.50, "unknown": 1.0,
}


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _settings(mode, priority, direction, bpm_tolerance):
    if mode not in {"fuzzy", "mood"}:
        raise ValueError("mode must be fuzzy or mood")
    priority = _number(priority, "priority")
    if not 0 <= priority <= 1:
        raise ValueError("priority must be 0 (BPM) to 1 (key)")
    if direction not in {"ascending", "descending", "none"}:
        raise ValueError("direction must be ascending, descending or none")
    if bpm_tolerance is not None:
        bpm_tolerance = _number(bpm_tolerance, "bpm_tolerance")
        if bpm_tolerance <= 0:
            raise ValueError("bpm_tolerance must be positive")
    return mode, priority, direction, bpm_tolerance


def _key(value):
    text = str(value or "").strip()
    text = _KEYS.get(text, text).upper()
    match = re.fullmatch(r"(1[0-2]|[1-9])([AB])", text)
    return (int(match[1]), match[2]) if match else None


def _track(record):
    identity = str(record.get("id") or record.get("path") or "")
    if not identity:
        raise ValueError("Every track requires an id or path")
    bpm = record.get("bpm")
    if bpm is not None:
        bpm = _number(bpm, f"{identity}.bpm")
        if bpm <= 0:
            raise ValueError("BPM must be positive")
    key = _key(record.get("key_camelot") or record.get("key"))
    return identity, bpm, key


def harmonic_relation(key_a, key_b, *, mode="fuzzy"):
    if mode not in {"fuzzy", "mood"}:
        raise ValueError("mode must be fuzzy or mood")
    a, b = _key(key_a), _key(key_b)
    relation, accepted = "unknown", False
    if a and b:
        step = (b[0] - a[0]) % 12
        same_letter = a[1] == b[1]
        if a == b:
            relation, accepted = "same_key", True
        elif step == 0:
            relation, accepted = "relative_major_minor", True
        elif same_letter and step in {1, 11}:
            relation, accepted = ("perfect_boost" if step == 1 else "perfect_drop"), True
        elif not same_letter and step in {1, 11}:
            relation, accepted = "diagonal", True
        elif mode == "mood" and same_letter and step == 2:
            relation, accepted = "energy_boost", True
        elif mode == "mood" and not same_letter and step == (3 if a[1] == "A" else 9):
            relation, accepted = "parallel_major_minor", True
        elif mode == "mood" and same_letter and step == 7:
            relation, accepted = "semitone_lift", True
        else:
            relation = "incompatible"
    return {"relation": relation, "accepted": accepted,
            "cost": RELATION_COSTS[relation],
            "key_from": f"{a[0]}{a[1]}" if a else None,
            "key_to": f"{b[0]}{b[1]}" if b else None,
            "risk": "high" if relation == "semitone_lift" else "review" if relation in {"diagonal", "energy_boost", "parallel_major_minor", "incompatible", "unknown"} else "lower",
            "acceptance_scope": "allowed by our explicit relation table; not a guarantee of compatible audio"}


def score_pair(a: Mapping[str, Any], b: Mapping[str, Any], *, mode="fuzzy",
               priority=.5, direction="ascending", bpm_tolerance=None):
    mode, priority, direction, bpm_tolerance = _settings(mode, priority, direction, bpm_tolerance)
    identity_a, bpm_a, _ = _track(a)
    identity_b, bpm_b, _ = _track(b)
    harmonic = harmonic_relation(a.get("key_camelot") or a.get("key"),
                                 b.get("key_camelot") or b.get("key"), mode=mode)
    known = bpm_a is not None and bpm_b is not None
    delta = bpm_b - bpm_a if known else None
    jump = abs(delta) if known else None
    reversal = (max(0., -delta) if direction == "ascending" else
                max(0., delta) if direction == "descending" else 0.) if known else None
    excess = max(0., jump - bpm_tolerance) if known and bpm_tolerance is not None else 0.
    tempo_weight = 2 * (1 - priority)
    components = {"harmonic": 2 * priority * harmonic["cost"],
                  "tempo": tempo_weight * (jump / 10 if known else 1.),
                  "direction": tempo_weight * .5 * (reversal / 10 if known else 0.),
                  "tolerance": excess / bpm_tolerance if bpm_tolerance else 0.}
    return {"from": identity_a, "to": identity_b, "cost": sum(components.values()),
            "components": components, "harmonic": harmonic,
            "harmonic_relation": harmonic["relation"],
            "tempo": {"delta_bpm": delta, "absolute_delta_bpm": jump,
                      "relative_change": max(bpm_a / bpm_b, bpm_b / bpm_a) - 1 if known else None,
                      "tolerance_excess_bpm": excess, "direction_reversal_bpm": reversal},
            "reasons": [harmonic["relation"], "BPM/key only; no energy or voice score",
                        "tempo unknown" if not known else f"tempo delta {delta:+.3f} BPM"],
            "transition_policy": "sync", "feasible": True,
            "feasibility_scope": "order score only; audio grid and temporary tempo rates not validated"}


def summarize_order(records, order, **settings):
    by_id = {str(r.get("id") or r.get("path")): r for r in records}
    if len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id):
        raise ValueError("Order must contain every input id exactly once")
    edges = [score_pair(by_id[a], by_id[b], **settings) for a, b in zip(order, order[1:])]
    known = [e["tempo"]["absolute_delta_bpm"] for e in edges if e["tempo"]["absolute_delta_bpm"] is not None]
    return {"total_cost": sum(e["cost"] for e in edges), "edges": edges,
            "harmonic_matches": sum(e["harmonic"]["accepted"] for e in edges),
            "lower_risk_matches": sum(e["harmonic"]["risk"] == "lower" for e in edges),
            "adventurous_matches": sum(e["harmonic"]["accepted"] and e["harmonic"]["risk"] != "lower" for e in edges),
            "harmonic_unknown": sum(e["harmonic"]["relation"] == "unknown" for e in edges),
            "maximum_bpm_jump": max(known, default=None), "total_bpm_travel": sum(known),
            "direction_reversals": sum((e["tempo"]["direction_reversal_bpm"] or 0) > 0 for e in edges),
            "tolerance_exceedances": sum(e["tempo"]["tolerance_excess_bpm"] > 0 for e in edges)}


def _relocate_blocks(order, costs, first, last, *, passes=8):
    """Improve directed paths by moving an indivisible block, preserving ends.

    Unlike a segment reversal this move leaves the relative order of all other
    blocks unchanged. Recompute the complete directed objective for each move:
    symmetric edge shortcuts would be incorrect for the Mood/direction rules.
    No positional costs enter the Harmonize objective. A finite pass cap keeps
    the additional search bounded, and only strict improvements are accepted.
    """
    path = list(order)
    nodes = [[0.] * len(path) for _ in path]
    start_bound = 1 if first is not None else 0
    end_bound = len(path) - 1 if last is not None else len(path)
    for _ in range(passes):
        old_cost = _order_cost(path, costs, nodes)
        best_path, best_cost = path, old_cost
        for source in range(start_bound, end_bound):
            without = path[:source] + path[source + 1:]
            for destination in range(start_bound, end_bound):
                if destination == source:
                    continue
                candidate = (without[:destination] + [path[source]]
                             + without[destination:])
                value = _order_cost(candidate, costs, nodes)
                if value < best_cost - 1e-12:
                    best_path, best_cost = candidate, value
        if best_cost >= old_cost - 1e-12:
            break
        path = best_path
    return path


def _search_matrix(pairs, order_objective):
    """Encode declared priorities without exchanging a bad key for a small BPM gain.

    At n blocks the complete normalized tertiary score is at most n-1.
    A review edge costs n, and an unaccepted edge costs n**2. Therefore
    neither lower tier can compensate for one extra violation of the tier
    above. Internal edges in locked chains are constant for every route.
    """
    if order_objective not in {"compatible_first", "weighted"}:
        raise ValueError("order_objective must be compatible_first or weighted")
    n = len(pairs)
    raw = [[pair["cost"] for pair in row] for row in pairs]
    if any(not math.isfinite(cost) or cost < 0 for row in raw for cost in row):
        raise ValueError("Pair costs must remain finite and nonnegative")
    if order_objective == "weighted":
        return raw, {"strategy": order_objective, "weighted_scale": 1.0}
    scale = max(1., max((raw[i][j] for i in range(n) for j in range(n) if i != j), default=0.))
    matrix = [[n*n * (not pair["harmonic"]["accepted"])
               + n * (pair["harmonic"]["accepted"] and pair["harmonic"]["risk"] != "lower")
               + pair["cost"] / scale for pair in row] for row in pairs]
    return matrix, {"strategy": order_objective, "weighted_scale": scale,
                    "unaccepted_weight": n*n, "accepted_review_weight": n,
                    "priority_order": ["fewest unaccepted/unknown key edges",
                                       "fewest accepted review/high-risk key edges",
                                       "lowest declared weighted BPM/key cost"]}


def harmonize_tracks(records: Sequence[Mapping[str, Any]], *, mode="fuzzy", priority=.5,
                     direction="ascending", bpm_tolerance=None, first_id=None,
                     last_id=None, locked_chains=None, seed=0, exact_limit=17,
                     beam_width=512, order_objective="compatible_first", search_seconds=20.0):
    """Retain every track once; locked adjacent chains move as indivisible blocks.

    Direction and BPM tolerance are soft penalties, never silently broken hard
    constraints. Unknown inputs remain explicit. Seed breaks equal-score ties.
    The returned order plan can be passed to the existing SYNC section selector.
    """
    mode, priority, direction, bpm_tolerance = _settings(mode, priority, direction, bpm_tolerance)
    if not records:
        raise ValueError("At least one track is required")
    if not isinstance(exact_limit, int) or not 1 <= exact_limit <= 18:
        raise ValueError("exact_limit must be between 1 and 18")
    if not isinstance(beam_width, int) or not 1 <= beam_width <= 5000:
        raise ValueError("beam_width must be between 1 and 5000")
    search_seconds = _number(search_seconds, "search_seconds")
    if not 0 < search_seconds <= 120:
        raise ValueError("search_seconds must be positive and at most 120")
    if order_objective not in {"compatible_first", "weighted"}:
        raise ValueError("order_objective must be compatible_first or weighted")
    by_id = {}
    paths = set()
    warnings = []
    for r in records:
        identity, bpm, key = _track(r)
        if identity in by_id:
            raise ValueError("Duplicate track ids are not allowed")
        path = r.get("path")
        if path and path in paths:
            raise ValueError("Duplicate source paths are not allowed")
        if path:
            paths.add(path)
        by_id[identity] = r
        if bpm is None or key is None:
            warnings.append(f"{identity}: missing BPM or supported musical key")
        if any(r.get(k) is not None and _number(r[k], k) < .65 for k in ("key_confidence", "bpm_confidence")):
            warnings.append(f"{identity}: uncertain analysis; confidence is reported but not used in the BPM/key objective")
    groups, assigned = [], set()
    for chain in locked_chains or []:
        if isinstance(chain, str) or not chain:
            raise ValueError("A locked chain must be a nonempty list of ids")
        chain = list(chain)
        if len(set(chain)) != len(chain) or set(chain) & assigned or not set(chain) <= set(by_id):
            raise ValueError("Locked chains require disjoint known ids without duplicates")
        assigned.update(chain)
        groups.append(chain)
    groups.extend([identity] for identity in by_id if identity not in assigned)
    blocks = {identity: i for i, chain in enumerate(groups) for identity in chain}
    for identity, start in ((first_id, True), (last_id, False)):
        if identity is None:
            continue
        if identity not in by_id:
            raise ValueError(f"Unknown locked endpoint: {identity}")
        if identity != groups[blocks[identity]][0 if start else -1]:
            raise ValueError("Locked endpoint conflicts with a locked chain")
    first, last = blocks.get(first_id), blocks.get(last_id)
    if first is not None and first == last and len(groups) > 1:
        raise ValueError("Different endpoint blocks are required")
    settings = dict(mode=mode, priority=priority, direction=direction, bpm_tolerance=bpm_tolerance)
    pairs = [[score_pair(by_id[a[-1]], by_id[b[0]], **settings) for b in groups] for a in groups]
    costs, objective_details = _search_matrix(pairs, order_objective)
    nodes = [[0.] * len(groups) for _ in groups]
    tied = list(range(len(groups)))
    random.Random(seed).shuffle(tied)
    ranks = {i: p for p, i in enumerate(tied)}
    baseline = _greedy(costs, nodes, first, last, ranks)
    solver = None
    if len(groups) <= exact_limit:
        chosen = _exact(costs, nodes, first, last, ranks)
        method, optimal = "exact Held-Karp over locked blocks", True
    else:
        chosen = _beam(costs, nodes, first, last, ranks, beam_width)
        if baseline is not None and _order_cost(baseline, costs, nodes) < _order_cost(chosen, costs, nodes):
            chosen = baseline
        chosen = _relocate_blocks(chosen, costs, first, last)
        method, optimal = f"bounded beam width {beam_width}, directional segment and relocation improvement", False
        if order_objective == "compatible_first" and len(groups) <= 100:
            from .harmonize_route import solve_hamiltonian_path
            solver = solve_hamiltonian_path(costs, first=first, last=last, time_limit=search_seconds)
            if solver is not None and solver["cost"] <= _order_cost(chosen, costs, nodes) + 1e-8:
                chosen = solver["order"]
                method = "directed whole-path MILP over locked blocks; incumbent verified independently"
                optimal = solver["optimal_for_objective"]
            elif solver is None:
                warnings.append("Whole-path solver found no valid incumbent within the budget; bounded search retained")
        elif order_objective == "compatible_first":
            warnings.append("More than 100 blocks: bounded search used; whole-path optimality not established")
    order = [identity for i in chosen for identity in groups[i]]
    result = summarize_order(records, order, **settings)
    baseline_order = [identity for i in baseline for identity in groups[i]]
    baseline_summary = summarize_order(records, baseline_order, **settings)
    baseline_cost = baseline_summary["total_cost"]
    search_cost = _order_cost(chosen, costs, nodes)
    baseline_search_cost = _order_cost(baseline, costs, nodes)
    for position, edge in enumerate(result["edges"], 1):
        edge["position"] = position
        edge["locked_adjacency"] = any((edge["from"], edge["to"]) in set(zip(c, c[1:])) for c in groups)
    result.update(schema_version=1, version=VERSION if order_objective != "weighted" else "superdj-harmonize-2", mode="harmonize", method=method,
        optimal_for_declared_objective=optimal, order=order,
        ordered_paths=[by_id[i].get("path") for i in order], settings=settings,
        relation_costs=dict(RELATION_COSTS), baseline_order=baseline_order,
        baseline_cost=baseline_cost, improvement=baseline_cost-result["total_cost"],
        improvement_scope="legacy weighted score only; can be negative when compatibility takes priority",
        order_objective=order_objective, objective_details=objective_details,
        search_cost=search_cost, baseline_search_cost=baseline_search_cost,
        search_cost_scope="block-boundary objective; comparable only with identical group partition and scaling",
        search_improvement=baseline_search_cost-search_cost,
        objective_tuple=[len(result['edges'])-result['harmonic_matches'], result['adventurous_matches'], result['total_cost']],
        baseline_objective_tuple=[len(baseline_summary['edges'])-baseline_summary['harmonic_matches'],
                                  baseline_summary['adventurous_matches'], baseline_summary['total_cost']],
        solver=solver, search_seconds=search_seconds,
        objective=("fewest unaccepted key edges, then fewest accepted review/high-risk edges, then weighted BPM/key cost"
                   if order_objective == "compatible_first" else
                   "2*priority*harmonic + 2*(1-priority)*(abs(deltaBPM)/10 + .5*direction_reversalBPM/10) + tolerance_excess/tolerance"),
        constraints={"first_id": first_id, "last_id": last_id, "locked_chains": locked_chains or [],
                     "all_tracks_once": True, "seed": seed, "direction_kind": "soft preference",
                     "bpm_tolerance_kind": "soft penalty; violations reported"},
        uncertainty={"warnings": warnings},
        render_readiness="order only; run grid, phrase, cue and temporary-tempo checks before SYNC rendering",
        proprietary_equivalence=False,
        official_sources=["https://help.dj.studio/en/articles/7878402-harmonize-previously-automix",
                          "https://help.dj.studio/en/articles/8118455-harmonic-matches-explained",
                          "https://dj.studio/blog/fuzzy-keymixing"],
        limitations=["Exact DJ.Studio weights and search are not published; this objective is ours",
                     "Official Fuzzy/Mood descriptions conflict; our rules are explicit experimental profiles",
                     "Same order is not required for a good result; different analysis and tie breaks change it",
                     "No automatic key shifting, BPM octave correction or library-track insertion",
                     "Native detector disagreements are not repaired by changing the route search",
                     "MILP time-limited incumbents are not optimality proofs; equal optima may have different orders"])
    return result
