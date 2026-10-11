"""Assemble an auditable playlist with a trained remaining-pool mode ranker.

This sequential policy recomputes musical connectivity after each choice. It
does not claim the global optimum of the separate pair-cost route optimizer.
"""
from __future__ import annotations

import random

import numpy as np

from .harmonize import _settings, _track, harmonic_relation, summarize_order
from .harmonize_modes import rank_mode_candidates, validate_mode_ranker_profile
from .harmonize_modes_repair import repair_mode_order


TAIL_GUARD_LIMIT = 12


def _minimum_forbidden_tail(forbidden, *, last=None):
    """Exact unrestricted suffix bridge minimum for every fixed first.

    Rows leave a block's tail; columns enter another block's head. Locked
    internal links are constant across all block permutations and excluded.
    The bound permits a penalized forbidden bridge even when another allowed
    head exists; the executing policy below still prefers an allowed head.
    A fixed last block is visited only as the final block. ``None`` means a
    first choice conflicts with that endpoint, not an infeasible music file.
    """
    values = np.asarray(forbidden)
    if (values.ndim != 2 or values.shape[0] != values.shape[1]
            or not 1 <= len(values) <= TAIL_GUARD_LIMIT):
        raise ValueError("Tail guard requires a square matrix of 1..12 blocks")
    if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
        raise ValueError("Tail guard forbidden values must be finite zero or one")
    count = len(values)
    if last is not None and (type(last) is not int or not 0 <= last < count):
        raise ValueError("Tail guard last index is outside the block matrix")
    bad = values.astype(bool)
    size = 1 << count
    dynamic = np.full((size, count), 255, dtype=np.uint8)
    members = [tuple(i for i in range(count) if mask & (1 << i)) for mask in range(size)]
    for index in range(count):
        if last is None or index == last:
            dynamic[1 << index, index] = 0
    for mask in range(1, size):
        if mask.bit_count() == 1 or last is not None and not mask & (1 << last):
            continue
        for first in members[mask]:
            if first == last:
                continue
            remaining = mask ^ (1 << first)
            best = min(int(bad[first, successor]) + int(dynamic[remaining, successor])
                       for successor in members[remaining])
            # 255 is unreachable; never wrap it to zero in an unsigned array.
            if best < 255:
                dynamic[mask, first] = best
    return [int(value) if value < 255 else None for value in dynamic[-1]]


def _tail_guard(remaining, blocks, by_id, mode, last):
    if len(remaining) > TAIL_GUARD_LIMIT:
        return None
    def allowed(left, right):
        return harmonic_relation(by_id[left].get("key_camelot") or by_id[left].get("key"),
            by_id[right].get("key_camelot") or by_id[right].get("key"), mode=mode)["accepted"]
    forbidden = [[not allowed(blocks[source][-1], blocks[target][0])
                  for target in remaining] for source in remaining]
    fixed_last = remaining.index(last) if last in remaining else None
    costs = _minimum_forbidden_tail(forbidden, last=fixed_last)
    internal = sum(not allowed(a, b) for index in remaining for a, b in zip(blocks[index], blocks[index][1:]))
    return dict(costs=dict(zip(remaining, costs)), internal_forbidden=int(internal),
                last_head=blocks[last][0] if last in remaining else None)


def sequence_mode_tracks(records, profile, *, mode="fuzzy", first_id=None,
                         last_id=None, locked_chains=None, seed=0,
                         priority=.5, direction="ascending", bpm_tolerance=None):
    """Retain every input/lock; explicitly flag unavoidable out-of-mode links.

    Learned utility only ranks allowed block heads. With at most 12 remaining
    blocks, exact unrestricted suffix bounds first minimize the number of later
    out-of-mode links conditional on the candidate first block, then keep
    current learned utility and seeded ties. This is not an optimum of the
    executing policy, which continues to prefer allowed heads. If none passes the mode
    gate, a declared BPM/key fallback retains the music and records a review
    requirement. A forbidden head is never called a learned recommendation.
    """
    profile = validate_mode_ranker_profile(profile)
    settings = dict(zip(("mode", "priority", "direction", "bpm_tolerance"),
                        _settings(mode, priority, direction, bpm_tolerance)))
    records = list(records)
    if not records:
        raise ValueError("At least one track is required")
    by_id, paths = {}, set()
    for record in records:
        identity, bpm, _ = _track(record)
        if identity in by_id or record.get("path") and record["path"] in paths:
            raise ValueError("Duplicate track IDs or source paths are not allowed")
        if bpm is None:
            raise ValueError("Mode sequencing requires positive BPM for every input")
        by_id[identity] = record
        if record.get("path"):
            paths.add(record["path"])
    blocks, assigned = [], set()
    for chain in locked_chains or []:
        if isinstance(chain, str) or not chain:
            raise ValueError("Locked chains require nonempty ID lists")
        chain = list(chain)
        if len(set(chain)) != len(chain) or set(chain) & assigned or not set(chain) <= set(by_id):
            raise ValueError("Locked chains require disjoint known IDs")
        blocks.append(chain)
        assigned.update(chain)
    blocks += [[identity] for identity in by_id if identity not in assigned]
    owners = {identity: index for index, block in enumerate(blocks) for identity in block}
    for identity, offset in ((first_id, 0), (last_id, -1)):
        if identity is not None and (identity not in owners or blocks[owners[identity]][offset] != identity):
            raise ValueError("Locked endpoint is unknown or conflicts with its chain")
    first, last = owners.get(first_id), owners.get(last_id)
    if first is not None and first == last and len(blocks) > 1:
        raise ValueError("Different endpoint blocks are required")
    permutation = list(range(len(blocks)))
    random.Random(seed).shuffle(permutation)
    tie_order = {index: position for position, index in enumerate(permutation)}
    if first is None:
        possible = [index for index in range(len(blocks)) if index != last or len(blocks) == 1]
        if direction != "none":
            sign = 1 if direction == "ascending" else -1
            first = min(possible, key=lambda index: (sign * _track(by_id[blocks[index][0]])[1], tie_order[index]))
        else:
            # Choose a connected starting key without using demonstrated order.
            def start_score(index):
                key = by_id[blocks[index][0]].get("key_camelot") or by_id[blocks[index][0]].get("key")
                degree = sum(harmonic_relation(key, by_id[block[0]].get("key_camelot") or by_id[block[0]].get("key"), mode=mode)["accepted"]
                             for other, block in enumerate(blocks) if other != index)
                return -degree, tie_order[index]
            first = min(possible, key=start_score)
    order, choices, warnings = list(blocks[first]), [], []
    remaining = [index for index in range(len(blocks)) if index != first]
    while remaining:
        available = [index for index in remaining if index != last or len(remaining) == 1]
        current = by_id[order[-1]]
        # All unplayed music contributes context; locks restrict playable heads.
        pool = [by_id[identity] for index in remaining for identity in blocks[index]]
        ranked = rank_mode_candidates(current, pool, profile, mode=mode)
        rows = {row["id"]: row for row in ranked["candidates"]}
        heads = [index for index in available if rows[blocks[index][0]]["eligible"]]
        fallback = not heads
        guard = _tail_guard(remaining, blocks, by_id, mode, last)
        finalists = heads if heads else available
        if guard is not None:
            minimum = min(guard["costs"][index] for index in finalists)
            finalists = [index for index in finalists if guard["costs"][index] == minimum]
        if fallback:
            from .harmonize import score_pair
            chosen = min(finalists, key=lambda index: (score_pair(current, by_id[blocks[index][0]], **settings)["cost"], tie_order[index]))
            warnings.append(f"{order[-1]} -> {blocks[chosen][0]}: no remaining playable head passes the harmonic mode; explicit fallback needs review")
        else:
            highest = max(rows[blocks[index][0]]["utility"] for index in finalists)
            tied = [index for index in finalists if highest - rows[blocks[index][0]]["utility"] <= 1e-8]
            chosen = min(tied, key=lambda index: tie_order[index])
        head = blocks[chosen][0]
        guard_evidence = dict(applied=guard is not None, remaining_block_count=len(remaining), exact_limit=TAIL_GUARD_LIMIT,
            scope="exact unrestricted suffix bridge lower bound conditional on first block and fixed last; execution still prefers an allowed head; not a policy or whole-route optimum")
        if guard is not None:
            guard_evidence.update(minimum_later_forbidden_among_considered_heads=minimum,
                considered_head_scope="allowed playable heads" if heads else "all playable fallback heads",
                chosen_minimum_later_forbidden=guard["costs"][chosen],
                remaining_locked_internal_forbidden=guard["internal_forbidden"],
                chosen_minimum_total_forbidden_remaining=guard["costs"][chosen] + guard["internal_forbidden"] + int(fallback),
                fixed_last_head=guard["last_head"],
                candidates=[dict(head=blocks[index][0], tail=blocks[index][-1], eligible_now=rows[blocks[index][0]]["eligible"],
                    playable_now=index in available, minimum_later_forbidden=guard["costs"][index]) for index in remaining])
        choices.append(dict(current=order[-1], chosen=head, candidate_count=len(pool),
                            playable_head_count=len(available), eligible_head_count=len(heads),
                            utility=rows[head]["utility"], fallback=fallback, tail_guard=guard_evidence,
                            choice_scope="explicit suffix-bound then BPM/key fallback" if fallback and guard is not None else
                                "explicit BPM/key fallback" if fallback else
                                "suffix-bound then learned remaining-pool utility among allowed playable heads" if guard is not None else
                                "learned remaining-pool utility among allowed playable heads"))
        order.extend(blocks[chosen])
        remaining.remove(chosen)
    incumbent_order, incumbent_warnings = list(order), list(warnings)
    repair = repair_mode_order(records, order, **settings,
        first_id=blocks[first][0], last_id=last_id, locked_chains=locked_chains)
    order = repair["order"]
    summary = summarize_order(records, order, **settings)
    warnings = []
    locked_pairs = {pair for chain in blocks for pair in zip(chain, chain[1:])}
    for position, edge in enumerate(summary["edges"], 1):
        edge.update(position=position, locked_adjacency=(edge["from"], edge["to"]) in locked_pairs)
        if edge["locked_adjacency"] and not edge["harmonic"]["accepted"]:
            warnings.append(f"{edge['from']} -> {edge['to']}: preserved locked adjacency is outside the harmonic mode")
        elif not edge["harmonic"]["accepted"]:
            warnings.append(f"{edge['from']} -> {edge['to']}: final route remains outside the harmonic mode after bounded search; explicit fallback bridge needs review")
    summary.update(schema_version=1, version="superdj-mode-sequence-3", mode="harmonize",
        method="sequential mode-aware remaining-pool utility; independent harmonic gate; exact small-tail guard; bounded directed complete-route repair",
        order=order, ordered_paths=[by_id[identity].get("path") for identity in order], settings=settings,
        optimal_for_declared_objective=False, order_objective="mode_pool",
        objective_details=dict(strategy="mode_pool", feature_family=profile["feature_family"], sharing=profile["sharing"],
            model_version=profile["model_version"], utility_warning=profile["utility_warning"],
            tail_guard=dict(exact_limit_blocks=TAIL_GUARD_LIMIT,
                priority="minimum later forbidden inter-block links, then current learned utility or declared fallback, then seeded tie",
                scope="exact unrestricted suffix lower bounds conditional on first and fixed last; executing allowed-head policy may not attain them; no whole-route/global-optimum claim"),
            route_repair=dict(version=repair["version"], changed=repair["changed"],
                objective=repair["objective"], candidate_budget=repair["candidate_budget"],
                scope=repair["scope"]),
            scope="sequential policy plus bounded whole-route repair; not a complete-route minimum or DJ.Studio implementation"),
        control_scope=dict(priority="diagnostic summary, explicit fallback and static complete-route repair tie cost; not fitted utility",
            bpm_tolerance="diagnostic summary, explicit fallback and static route-repair tie cost; not a learned-candidate constraint",
            direction="automatic first choice, diagnostic summary, explicit fallback and static route-repair tie cost",
            learned_choice="musical utilities learned from supplied demonstrations; independent harmonic mode gate"),
        constraints=dict(first_id=first_id, last_id=last_id, locked_chains=locked_chains or [],
                         all_tracks_once=True, seed=seed),
        choices=choices, incumbent_order=incumbent_order,
        choices_scope="initial sequential decisions on incumbent_order and its actual remaining pools; not decisions on a repaired final order",
        incumbent_fallback_count=sum(choice["fallback"] for choice in choices),
        incumbent_uncertainty=dict(warnings=incumbent_warnings), route_repair=repair,
        fallback_count=sum(not edge["harmonic"]["accepted"] and not edge["locked_adjacency"] for edge in summary["edges"]),
        fallback_count_scope="unlocked final route bridges outside the mode; separate from initial sequential fallback decisions",
        initial_choice_scope="locked first" if first_id is not None else "BPM direction or harmonic connectivity heuristic",
        uncertainty=dict(warnings=warnings), proprietary_equivalence=False,
        render_readiness="order only; validate source, grid, cues and temporary tempo before fixed-order SYNC rendering")
    return summary
