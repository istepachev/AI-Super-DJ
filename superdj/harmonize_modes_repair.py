"""Bounded directed repair of a learned incumbent, preserving user locks.

This is a separate complete-route safety pass. It neither changes next-track
rankings nor sums query-dependent model utilities. Only a strict reduction in
unaccepted boundary edges can alter an incumbent; compatible routes stay put.
"""
from __future__ import annotations

import math
import time

from .harmonize import _settings, _track, score_pair, summarize_order


VERSION = "superdj-mode-route-repair-2"


def _edges(path):
    return set(zip(path, path[1:]))


def _repair_route(order, blocked, costs, *, first=None, last=None, passes=8,
                  max_candidates=200_000, max_scans=None,
                  segment_lengths=(1, 2, 3)):
    """Relocate consecutive blocks or swap two, without reversing them.

    A move changes at most six directed boundary edges. The objective is, in
    order: fewer unaccepted edges, fewer changed incumbent adjacencies, lower
    static boundary cost. Objective evaluations and cheap scans have separate
    finite budgets. A strict improvement must remove a current unaccepted edge;
    skipping moves without one cannot discard an improving candidate.
    """
    path = list(order)
    original = _edges(path)
    n = len(path)
    start = 1 if first is not None else 0
    end = n - 1 if last is not None else n
    if first is not None and path[0] != first:
        raise ValueError("Locked first must already be first")
    if last is not None and path[-1] != last:
        raise ValueError("Locked last must already be last")
    max_scans = min(4_000_000, 20 * max_candidates) if max_scans is None else max_scans
    evaluations, scans, accepted = 0, 0, []
    local_plateau = False
    bad = sum(int(blocked[a][b]) for a, b in zip(path, path[1:]))
    total = sum(costs[a][b] for a, b in zip(path, path[1:]))
    retained = len(original)

    def delta(removed, added):
        return (sum(int(blocked[a][b]) for a, b in added)
                - sum(int(blocked[a][b]) for a, b in removed),
                sum((a, b) in original for a, b in added)
                - sum((a, b) in original for a, b in removed),
                sum(costs[a][b] for a, b in added)
                - sum(costs[a][b] for a, b in removed))

    for _ in range(passes):
        if bad == 0 or evaluations >= max_candidates or scans >= max_scans:
            break
        best, best_key = None, None
        for length in segment_lengths:
            if length >= n:
                continue
            for source in range(start, end - length + 1):
                segment = path[source:source + length]
                before = path[source - 1] if source else None
                after = path[source + length] if source + length < n else None
                removed0, added0 = [], []
                if before is not None:
                    removed0.append((before, segment[0]))
                if after is not None:
                    removed0.append((segment[-1], after))
                if before is not None and after is not None:
                    added0.append((before, after))
                without = path[:source] + path[source + length:]
                destination_end = len(without) if last is None else len(without) - 1
                for destination in range(start, destination_end + 1):
                    if destination == source:
                        continue
                    if evaluations >= max_candidates or scans >= max_scans:
                        break
                    scans += 1
                    left = without[destination - 1] if destination else None
                    right = without[destination] if destination < len(without) else None
                    removed, added = list(removed0), list(added0)
                    if left is not None and right is not None:
                        removed.append((left, right))
                    if left is not None:
                        added.append((left, segment[0]))
                    if right is not None:
                        added.append((segment[-1], right))
                    if not any(blocked[a][b] for a, b in removed):
                        continue
                    evaluations += 1
                    dbad, dkeep, dcost = delta(removed, added)
                    if dbad >= 0:
                        continue
                    key = (bad + dbad, -(retained + dkeep), total + dcost)
                    if best_key is None or key < best_key:
                        candidate = without[:destination] + segment + without[destination:]
                        best, best_key = (candidate, dbad, dkeep, dcost,
                            dict(move="relocate", source_block_index=source,
                                 destination_block_index=destination, length_blocks=length)), key
                if evaluations >= max_candidates or scans >= max_scans:
                    break
            if evaluations >= max_candidates or scans >= max_scans:
                break
        if evaluations < max_candidates and scans < max_scans:
            for left in range(start, end):
                for right in range(left + 1, end):
                    if evaluations >= max_candidates or scans >= max_scans:
                        break
                    scans += 1
                    indices = sorted(i for i in {left - 1, left, right - 1, right}
                                     if 0 <= i < n - 1)

                    def moved(i):
                        return path[right] if i == left else path[left] if i == right else path[i]

                    removed = [(path[i], path[i + 1]) for i in indices]
                    added = [(moved(i), moved(i + 1)) for i in indices]
                    if not any(blocked[a][b] for a, b in removed):
                        continue
                    evaluations += 1
                    dbad, dkeep, dcost = delta(removed, added)
                    if dbad >= 0:
                        continue
                    key = (bad + dbad, -(retained + dkeep), total + dcost)
                    if best_key is None or key < best_key:
                        candidate = list(path)
                        candidate[left], candidate[right] = candidate[right], candidate[left]
                        best, best_key = (candidate, dbad, dkeep, dcost,
                            dict(move="swap", left_block_index=left, right_block_index=right)), key
                if evaluations >= max_candidates or scans >= max_scans:
                    break
        if best is None:
            local_plateau = evaluations < max_candidates and scans < max_scans
            break
        path, dbad, dkeep, dcost, description = best
        bad, retained, total = bad + dbad, retained + dkeep, total + dcost
        description.update(unaccepted_boundaries_after=bad,
                           incumbent_block_edges_retained=retained)
        accepted.append(description)
        # Independently verify constant-size delta calculations after mutation.
        actual_bad = sum(int(blocked[a][b]) for a, b in zip(path, path[1:]))
        if (actual_bad != bad or len(_edges(path) & original) != retained
                or sorted(path) != sorted(order)
                or (first is not None and path[0] != first)
                or (last is not None and path[-1] != last)):
            raise RuntimeError("Directed route repair failed independent verification")
    return path, dict(evaluated_candidates=evaluations, scanned_candidates=scans,
                      pruned_candidates=scans - evaluations, accepted_moves=accepted,
                      incumbent_block_order=list(order), result_block_order=list(path),
                      segment_lengths=list(segment_lengths), scan_budget=max_scans,
                      passes_cap=passes, candidate_budget=max_candidates,
                      local_plateau=local_plateau,
                      stop_reason=("compatible boundaries" if bad == 0 else
                                   "budget exhausted" if evaluations >= max_candidates or scans >= max_scans else
                                   "strict improvement plateau" if local_plateau else "pass cap"),
                      budget_exhausted=evaluations >= max_candidates or scans >= max_scans)


def repair_mode_order(records, order, *, mode="fuzzy", first_id=None,
                      last_id=None, locked_chains=None, priority=.5,
                      direction="ascending", bpm_tolerance=None, passes=8,
                      max_candidates=200_000, boundary_costs=None, max_scans=None):
    """Repair a complete incumbent with fixed endpoints and adjacent chains.

    ``boundary_costs`` is an internal optional square matrix over blocks in
    incumbent order. It may supply a static musical boundary objective for
    tie-breaking; query-dependent learned utilities are unsuitable here.
    Musical arcs and audio feasibility are outside this safety pass's scope.
    """
    mode, priority, direction, bpm_tolerance = _settings(mode, priority, direction, bpm_tolerance)
    for value, label, limit in ((passes, "passes", 32),
                                (max_candidates, "max_candidates", 2_000_000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
            raise ValueError(f"{label} must be an integer from zero to {limit}")
    if max_scans is not None and (isinstance(max_scans, bool) or not isinstance(max_scans, int)
                                 or not 0 <= max_scans <= 20_000_000):
        raise ValueError("max_scans must be an integer from zero to 20000000")
    scan_budget = min(4_000_000, 20 * max_candidates) if max_scans is None else max_scans
    if not records:
        raise ValueError("At least one track is required")
    by_id, paths = {}, set()
    for record in records:
        identity, _, _ = _track(record)
        if identity in by_id:
            raise ValueError("Duplicate track ids are not allowed")
        path = record.get("path")
        if path and path in paths:
            raise ValueError("Duplicate source paths are not allowed")
        if path:
            paths.add(path)
        by_id[identity] = record
    order = list(order)
    if len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id):
        raise ValueError("Order must contain every input id exactly once")
    if first_id is not None and (first_id not in by_id or order[0] != first_id):
        raise ValueError("Locked first must already be first")
    if last_id is not None and (last_id not in by_id or order[-1] != last_id):
        raise ValueError("Locked last must already be last")
    chains, assigned = {}, set()
    for chain in locked_chains or []:
        if isinstance(chain, str) or not chain:
            raise ValueError("A locked chain must be a nonempty list of ids")
        chain = list(chain)
        if len(set(chain)) != len(chain) or assigned.intersection(chain) or not set(chain) <= set(by_id):
            raise ValueError("Locked chains require disjoint known ids without duplicates")
        position = order.index(chain[0])
        if order[position:position + len(chain)] != chain:
            raise ValueError("Incumbent must already preserve every locked adjacency")
        chains[chain[0]] = chain
        assigned.update(chain)
    groups, position = [], 0
    while position < len(order):
        group = chains.get(order[position], [order[position]])
        groups.append(group)
        position += len(group)
    first = 0 if first_id is not None else None
    last = len(groups) - 1 if last_id is not None else None
    settings = dict(mode=mode, priority=priority, direction=direction, bpm_tolerance=bpm_tolerance)
    pairs = [[score_pair(by_id[a[-1]], by_id[b[0]], **settings) for b in groups] for a in groups]
    blocked = [[not pair["harmonic"]["accepted"] for pair in row] for row in pairs]
    costs = [[pair["cost"] for pair in row] for row in pairs]
    if boundary_costs is not None:
        if len(boundary_costs) != len(groups) or any(len(row) != len(groups) for row in boundary_costs):
            raise ValueError("boundary_costs must be a square matrix over incumbent blocks")
        costs = [[float(value) for value in row] for row in boundary_costs]
        if any(not math.isfinite(value) for row in costs for value in row):
            raise ValueError("boundary_costs must be finite")
    original_blocks = list(range(len(groups)))
    boundary_before = sum(blocked[a][b] for a, b in zip(original_blocks, original_blocks[1:]))
    started = time.perf_counter()
    repaired, short = _repair_route(original_blocks, blocked, costs, first=first, last=last,
                                    passes=passes, max_candidates=max_candidates, max_scans=scan_budget)
    short["stage"] = "short segments and swaps"
    audits = [short]
    after_short = sum(blocked[a][b] for a, b in zip(repaired, repaired[1:]))
    remaining_candidates = max_candidates - short["evaluated_candidates"]
    remaining_scans = scan_budget - short["scanned_candidates"]
    if after_short and short["local_plateau"] and remaining_candidates > 0 and remaining_scans > 0:
        repaired, long = _repair_route(repaired, blocked, costs, first=first, last=last,
            passes=passes, max_candidates=remaining_candidates, max_scans=remaining_scans,
            segment_lengths=tuple(range(1, 33)))
        long["stage"] = "long segments after remaining unaccepted boundaries"
        audits.append(long)
    moves = [dict(move, stage=audit["stage"]) for audit in audits for move in audit["accepted_moves"]]
    audit = dict(evaluated_candidates=sum(stage["evaluated_candidates"] for stage in audits),
                 scanned_candidates=sum(stage["scanned_candidates"] for stage in audits),
                 pruned_candidates=sum(stage["pruned_candidates"] for stage in audits),
                 accepted_moves=moves, stages=audits, passes_cap=passes,
                 candidate_budget=max_candidates, scan_budget=scan_budget,
                 budget_exhausted=any(stage["budget_exhausted"] for stage in audits))
    result_order = [identity for group in repaired for identity in groups[group]]
    before = summarize_order(records, order, **settings)
    after = summarize_order(records, result_order, **settings)
    unaccepted_before = len(before["edges"]) - before["harmonic_matches"]
    unaccepted_after = len(after["edges"]) - after["harmonic_matches"]
    if (result_order != order and unaccepted_after >= unaccepted_before
            or first_id is not None and result_order[0] != first_id
            or last_id is not None and result_order[-1] != last_id):
        raise RuntimeError("Complete-route repair failed independent verification")
    original_edges = _edges(order)
    retained = len(original_edges & _edges(result_order))
    return dict(schema_version=1, version=VERSION, order=result_order, incumbent_order=order,
                changed=result_order != order, all_inputs_once=True,
                constraints=dict(first_id=first_id, last_id=last_id,
                                 locked_chains=[list(chain) for chain in chains.values()]),
                unaccepted_before=unaccepted_before, unaccepted_after=unaccepted_after,
                unaccepted_boundaries_before=boundary_before,
                locked_unaccepted_constant=unaccepted_before - boundary_before,
                incumbent_track_edges_retained=retained,
                incumbent_track_edges_changed=len(order) - 1 - retained,
                elapsed_seconds=time.perf_counter() - started,
                boundary_cost_source="custom static matrix" if boundary_costs is not None else "declared BPM/key pair cost",
                optimal_for_declared_objective=False,
                objective=["strictly fewer unaccepted boundary edges",
                           "fewest changed stage-incumbent block adjacencies", "lowest static boundary cost"],
                scope="separate complete-route repair; next-track ranks unchanged; audio and global optimality unverified",
                **audit)
