"""Bounded complete-path search with dynamic mode-aware remaining-pool scores.

The beam keeps alternative prefixes through the final block, recomputing each
prefix's actual musical pool. Conditional log weights are normalized before
being added; raw utilities from different pools are never simply summed.
This is our experimental search, not a proprietary Harmonize implementation
or a proof of the best musical/complete route.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from .harmonize import _settings
from .harmonize_modes import (_allowed, _musical, _pair, _rank_pool_context, feature_names,
                              validate_mode_ranker_profile)
from .harmonize_ranker import _rank


VERSION = "superdj-mode-route-search-1"
LOSS_SCOPE = ("sum of per-step normalized fitted musical log weights; "
              "not calibrated confidence, human preference or sound quality")
POOL_FAMILIES = {"pool", "tempo_pool", "tempo_rank_density", "tempo_rank_graph"}
RANK_FAMILIES = {"tempo_rank_density", "tempo_rank_graph"}


def _integer(value, label, maximum=None):
    if type(value) is not int or value < 0 or maximum is not None and value > maximum:
        raise ValueError(f"{label} must be an integer in the declared search budget")
    return value


def _finite(value, label):
    if not math.isfinite(value):
        raise ValueError(f"{label} is non-finite")
    return float(value)


class DynamicModePool:
    """Exact public musical features with bounded caches, without ID features.

    Pair terms and the directed key table are reusable. Pool connectivity,
    tempo ranks/densities and continuation descriptors are recomputed for each
    actual unplayed mask, including every track inside a remaining locked
    block. Current-dependent eligible tempo ranks use that prefix's current
    track. Source IDs address records only; input positions provide seeded ties.
    """

    def __init__(self, records, profile, *, mode="fuzzy", seed=0, cache_limit=3000,
                 priority="unknown", bpm_tolerance=None):
        self.profile = validate_mode_ranker_profile(profile)
        from .harmonize_priority import effective_weights, routing_controls, tolerance_features
        self.routing_controls = routing_controls(priority, bpm_tolerance)
        if mode not in {"fuzzy", "mood"}:
            raise ValueError("mode must be fuzzy or mood")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.cache_limit = _integer(cache_limit, "cache_limit", 10000)
        self.records, self.mode = list(records), mode
        if not self.records:
            raise ValueError("At least one track is required")
        self.by_id, paths = {}, set()
        for index, record in enumerate(self.records):
            identity, _, _ = _musical(record)
            if identity in self.by_id or record.get("path") and record["path"] in paths:
                raise ValueError("Duplicate track IDs or source paths are not allowed")
            self.by_id[identity] = index
            if record.get("path"):
                paths.add(record["path"])
        self.count = len(self.records)
        self.bpms = np.array([_musical(record)[1] for record in self.records])
        self.keys = [_musical(record)[2] for record in self.records]
        self.family = self.profile["feature_family"]
        sharing = self.profile["sharing"]
        weights = np.array(self.profile["coefficients"][mode if sharing == "separate" else "all"])
        base_count = len(feature_names(self.family, "shared"))
        self.tolerance_weights = None
        if self.profile.get("priority_aware") is True:
            weights, self.tolerance_weights = effective_weights(self.profile, mode, **self.routing_controls)
        elif sharing == "interactions":
            with np.errstate(over="ignore", invalid="ignore"):
                weights = weights[:base_count] + float(mode == "mood") * weights[base_count:]
        if not np.all(np.isfinite(weights)):
            raise ValueError("Combined mode coefficients are non-finite")
        pair_count = len(_pair(self.records[0], self.records[0], mode, self.family))
        self.context_weights = weights[pair_count:pair_count + 12] if self.family in POOL_FAMILIES else None
        self.rank_weights = weights[pair_count + 12:pair_count + 31] if self.family in RANK_FAMILIES else None
        self.graph_weights = weights[pair_count + 31:pair_count + 36] if self.family == "tempo_rank_graph" else None
        self.allowed = np.array([[_allowed(a, b, mode) for b in self.keys] for a in self.keys])
        np.fill_diagonal(self.allowed, False)
        self.pair_utility = np.empty((self.count, self.count))
        self.relation_rank_utility = np.empty((self.count, self.count)) if self.rank_weights is not None else None
        for source, a in enumerate(self.records):
            for target, b in enumerate(self.records):
                values = _pair(a, b, mode, self.family)
                with np.errstate(over="ignore", invalid="ignore"):
                    self.pair_utility[source, target] = np.array(values) @ weights[:len(values)]
                    if self.tolerance_weights is not None:
                        terms = tolerance_features(self.bpms[source], self.bpms[target:target+1],
                            self.routing_controls["bpm_tolerance"])[0]
                        self.pair_utility[source, target] += terms @ self.tolerance_weights
                    if self.relation_rank_utility is not None:
                        self.relation_rank_utility[source, target] = np.array(values[:11]) @ self.rank_weights[8:]
        if not np.all(np.isfinite(self.pair_utility)):
            raise ValueError("Reused pair utility is non-finite")
        if self.relation_rank_utility is not None and not np.all(np.isfinite(self.relation_rank_utility)):
            raise ValueError("Reused relation tempo-rank utility is non-finite")
        self.same = np.array([[a == b for b in self.keys] for a in self.keys])
        self.gaps = np.abs(self.bpms[:, None] - self.bpms[None, :]) / 10
        self.tempo_delta = self.bpms[:, None] - self.bpms[None, :] if self.rank_weights is not None else None
        self.close2 = np.abs(self.tempo_delta) <= 2 if self.tempo_delta is not None else None
        if self.close2 is not None:
            np.fill_diagonal(self.close2, False)
        permutation = list(range(self.count))
        random.Random(seed).shuffle(permutation)
        self.tie = np.empty(self.count, dtype=int)
        self.tie[permutation] = np.arange(self.count)
        self.cache, self.rank_cache, self.calls, self.cache_hits = {}, {}, 0, 0

    def _context(self, mask):
        if mask in self.cache:
            self.cache_hits += 1
            return self.cache[mask]
        indices = np.array([i for i in range(self.count) if mask >> i & 1], dtype=int)
        count = len(indices)
        if not count:
            return indices, np.empty(0)
        if self.family not in POOL_FAMILIES:
            result = indices, np.zeros(count)
        else:
            adjacency = self.allowed[np.ix_(indices, indices)]
            outgoing, incoming = adjacency.sum(1), adjacency.sum(0)
            denominator = max(1, count - 1)
            same = self.same[np.ix_(indices, indices)].sum(1) - 1
            total_degree = outgoing + incoming
            residual = total_degree[None, :] - adjacency - adjacency.T
            stranded = ((residual == 0) & ~np.eye(count, dtype=bool)).sum(1)
            sole = adjacency[outgoing == 1].sum(0)
            nearest = np.minimum(np.where(adjacency, self.gaps[np.ix_(indices, indices)], np.inf).min(1), 3.)
            bpms = self.bpms[indices]
            matrix = np.column_stack([same / denominator, np.log1p(same),
                outgoing / denominator, incoming / denominator, 1 / (1 + outgoing),
                outgoing == 0, stranded / denominator, sole / denominator, nearest,
                np.abs(bpms - np.median(bpms)) / 10,
                bpms == bpms.min(), bpms == bpms.max()])
            with np.errstate(over="ignore", invalid="ignore"):
                result = indices, matrix @ self.context_weights
            if not np.all(np.isfinite(result[1])):
                raise ValueError("Actual pool context utility is non-finite")
        if self.cache_limit:
            if len(self.cache) >= self.cache_limit:
                self.cache.clear()
                self.rank_cache.clear()
            self.cache[mask] = result
        return result

    def _rank_context(self, current, mask, indices):
        """Exact rank-family columns without rebuilding Python key tables.

        Mask-only quantities can be reused. Allowed signed ranks, distance
        from current's signed rank and allowed-peer density must be recomputed
        for the actual current track. All peers retain their multiplicity.
        """
        cached = self.rank_cache.get(mask)
        if cached is None:
            bpms = self.bpms[indices]
            pool_columns = _rank_pool_context(bpms, [self.keys[index] for index in indices], self.mode,
                adjacency=self.allowed[np.ix_(indices, indices)], include_graph=self.graph_weights is not None)
            weights = self.rank_weights if self.graph_weights is None else np.concatenate((self.rank_weights, self.graph_weights))
            with np.errstate(over="ignore", invalid="ignore"):
                fixed_utility = pool_columns @ weights
            # The common feature helper defines mask-only columns. Cache
            # vectors only, not an n-by-n matrix or every feature column.
            cached = bpms, pool_columns[:, 0], fixed_utility
            if self.cache_limit:
                if len(self.rank_cache) >= self.cache_limit:
                    self.rank_cache.clear()
                self.rank_cache[mask] = cached
        bpms, rank, fixed_utility = cached
        eligible = self.allowed[current, indices]
        allowed_rank = np.zeros(len(indices))
        if eligible.any():
            allowed_delta = self.tempo_delta[np.ix_(indices[eligible], indices[eligible])]
            allowed_rank[eligible] = ((allowed_delta > .01).sum(1) - (allowed_delta < -.01).sum(1)) / max(1, int(eligible.sum()) - 1)
        current_delta = self.bpms[current] - bpms
        current_rank = ((current_delta > .01).sum() - (current_delta < -.01).sum()) / max(1, len(indices))
        rank_distance = np.abs(rank - current_rank) / 2
        allowed_density = self.close2[np.ix_(indices, indices[eligible])].sum(1) / max(1, int(eligible.sum()) - 1)
        columns = np.column_stack([allowed_rank, rank_distance, allowed_density])
        with np.errstate(over="ignore", invalid="ignore"):
            result = fixed_utility + columns @ self.rank_weights[[1, 4, 7]] + rank * self.relation_rank_utility[current, indices]
        if not np.all(np.isfinite(result)):
            raise ValueError("Actual remaining-pool tempo-rank utility is non-finite")
        return result

    def scores(self, current, mask):
        """Return actual-mask indices, utility and the independent mode gate."""
        if type(current) is not int or not 0 <= current < self.count:
            raise ValueError("Current index is outside the input records")
        if type(mask) is not int or mask < 0 or mask >= 1 << self.count or mask >> current & 1:
            raise ValueError("Unplayed mask must exclude current and contain only inputs")
        self.calls += 1
        indices, context = self._context(mask)
        with np.errstate(over="ignore", invalid="ignore"):
            utilities = self.pair_utility[current, indices] + context
            if self.rank_weights is not None and len(indices):
                utilities = utilities + self._rank_context(current, mask, indices)
        if not np.all(np.isfinite(utilities)):
            raise ValueError("Actual remaining-pool utility is non-finite")
        return indices, utilities, self.allowed[current, indices]

    def normalized_choices(self, current, mask, playable, *, ties=None):
        """Normalize among allowed playable heads, or disclose a BPM bridge.

        A head is a track index and a tie position. The fallback is native-BPM
        distance when no playable head passes the key gate; it is never called
        a learned harmonic recommendation. Locked internals are not choices.
        """
        indices, utility, eligible = self.scores(current, mask)
        positions = {int(index): i for i, index in enumerate(indices)}
        playable = list(playable)
        if not playable or any(index not in positions for index in playable):
            raise ValueError("Playable heads must be nonempty unplayed input indices")
        chosen = [index for index in playable if eligible[positions[index]]]
        fallback = not chosen
        if fallback:
            chosen = playable
            values = [-float(self.gaps[current, index]) for index in chosen]
        else:
            values = [float(utility[positions[index]]) for index in chosen]
        ranks, _ = _rank(values, 1e-8)
        tie = ties if ties is not None else self.tie
        ordering = sorted(range(len(chosen)), key=lambda i: (ranks[i], int(tie[chosen[i]])))
        maximum = max(values)
        logsum = _finite(math.log(sum(math.exp(value - maximum) for value in values)), "Playable-head normalizer")
        return [(chosen[i], _finite((values[i] - maximum) - logsum,
            "Normalized playable-head log weight"), fallback) for i in ordering]

    def rescore_order(self, order):
        """Diagnostic fit loss on every actual prefix, including forced links.

        Normalization uses all remaining candidates independently of the
        declared fitting support. This diagnostic is not a probability of the
        constrained block policy or an allowed-support fitting objective.
        It never selects between the local safety order and the beam order.
        """
        if len(order) != self.count or len(set(order)) != self.count or set(order) != set(self.by_id):
            raise ValueError("Order must contain every input exactly once")
        mask, loss, blocked = (1 << self.count) - 1, 0., 0
        for source, target in zip(order, order[1:]):
            current, successor = self.by_id[source], self.by_id[target]
            mask ^= 1 << current
            indices, utility, _ = self.scores(current, mask)
            position = int(np.flatnonzero(indices == successor)[0])
            maximum = float(utility.max())
            with np.errstate(over="ignore", invalid="ignore"):
                logsum = _finite(math.log(float(np.exp(utility - maximum).sum())), "Route normalizer")
            loss = _finite(loss + (maximum - float(utility[position])) + logsum, "Normalized route musical loss")
            blocked += not self.allowed[current, successor]
        return dict(normalized_musical_loss=float(loss), harmonic_violations=int(blocked),
                    loss_scope=LOSS_SCOPE,
                    training_support=self.profile.get("training_support", "all_candidates"),
                    normalization="all actual unplayed candidates per prefix, regardless of fitting support; forced links included; diagnostic only")


def search_mode_route(records, profile, *, mode="fuzzy", first_id, last_id=None,
                      locked_chains=None, seed=0, beam_width=32, branch_limit=7,
                      max_expansions=100000, cache_limit=3000,
                      priority="unknown", bpm_tolerance=None):
    """Search all the way to the final block with finite prefix/evaluation caps.

    Each locked chain is indivisible. The last block can be chosen only last.
    At a budget stop the best surviving prefix receives a declared greedy
    completion, retaining every source. Internal locked bad links are constant
    and excluded from prefix comparison; final scoring discloses them.
    """
    _settings(mode, .5, "none", None)
    if not 1 <= _integer(beam_width, "beam_width", 128):
        raise ValueError("beam_width must be positive")
    if not 1 <= _integer(branch_limit, "branch_limit", 32):
        raise ValueError("branch_limit must be positive")
    _integer(max_expansions, "max_expansions", 1000000)
    started = time.perf_counter()
    engine = DynamicModePool(records, profile, mode=mode, seed=seed, cache_limit=cache_limit,
        priority=priority, bpm_tolerance=bpm_tolerance)
    blocks, assigned = [], set()
    for chain in locked_chains or []:
        if isinstance(chain, str) or not chain:
            raise ValueError("Locked chains require nonempty ID lists")
        chain = list(chain)
        if len(set(chain)) != len(chain) or set(chain) & assigned or not set(chain) <= set(engine.by_id):
            raise ValueError("Locked chains require disjoint known IDs")
        blocks.append(chain)
        assigned.update(chain)
    blocks += [[identity] for identity in engine.by_id if identity not in assigned]
    owners = {identity: index for index, block in enumerate(blocks) for identity in block}
    if first_id not in owners or blocks[owners[first_id]][0] != first_id:
        raise ValueError("First endpoint is unknown or conflicts with its chain")
    if last_id is not None and (last_id not in owners or blocks[owners[last_id]][-1] != last_id):
        raise ValueError("Last endpoint is unknown or conflicts with its chain")
    first, last = owners[first_id], owners.get(last_id)
    if first == last and len(blocks) > 1:
        raise ValueError("Different endpoint blocks are required")
    heads = [engine.by_id[block[0]] for block in blocks]
    tails = [engine.by_id[block[-1]] for block in blocks]
    track_masks = [sum(1 << engine.by_id[identity] for identity in block) for block in blocks]
    head_owner = dict(zip(heads, range(len(blocks))))
    permutation = list(range(len(blocks)))
    random.Random(seed).shuffle(permutation)
    block_ties = {heads[index]: position for position, index in enumerate(permutation)}
    remaining_blocks = ((1 << len(blocks)) - 1) ^ (1 << first)
    remaining_tracks = ((1 << engine.count) - 1) ^ track_masks[first]
    # State: forbidden boundaries, normalized log weight, block path,
    # actual unplayed block and track masks.
    states = [(0, 0., (first,), remaining_blocks, remaining_tracks)]
    expansions, levels, exhausted, progress = 0, 0, False, []

    def choices(state):
        _, _, path, block_mask, track_mask = state
        available = [index for index in range(len(blocks)) if block_mask >> index & 1
                     and (index != last or block_mask.bit_count() == 1)]
        return engine.normalized_choices(tails[path[-1]], track_mask,
            [heads[index] for index in available], ties=block_ties)

    def state_key(state):
        return state[0], -round(state[1], 8), tuple(block_ties[heads[i]] for i in state[2])

    for _ in range(len(blocks) - 1):
        expanded = []
        for state in states:
            bad, weight, path, block_mask, track_mask = state
            for head, logweight, fallback in choices(state)[:branch_limit]:
                if expansions >= max_expansions:
                    exhausted = True
                    break
                block = head_owner[head]
                expanded.append((bad + int(fallback), _finite(weight + logweight, "Normalized prefix log weight"),
                    path + (block,), block_mask ^ (1 << block), track_mask ^ track_masks[block]))
                expansions += 1
            if exhausted:
                break
        if expanded:
            expanded.sort(key=state_key)
            seen, states = set(), []
            for state in expanded:
                key = state[2][-1], state[3]
                if key not in seen:
                    seen.add(key)
                    states.append(state)
                if len(states) >= beam_width:
                    break
            levels += 1
            progress.append(dict(completed_blocks=len(states[0][2]), surviving_prefixes=len(states),
                best_forbidden_boundaries=states[0][0], expansions=expansions))
        if exhausted or not expanded:
            exhausted = bool(remaining_blocks) if not expanded else exhausted
            break
    best = min(states, key=state_key)
    completion_count = best[3].bit_count()
    while best[3]:
        bad, weight, path, block_mask, track_mask = best
        head, logweight, fallback = choices(best)[0]
        block = head_owner[head]
        best = (bad + int(fallback), _finite(weight + logweight, "Completed prefix log weight"), path + (block,),
                block_mask ^ (1 << block), track_mask ^ track_masks[block])
    order = [identity for block in best[2] for identity in blocks[block]]
    final_loss = engine.rescore_order(order)
    assert order[0] == first_id and (last_id is None or order[-1] == last_id)
    assert len(order) == len(set(order)) == engine.count
    return dict(version=VERSION, order=order,
        method="bounded complete-prefix beam with actual remaining-pool normalization and directed mode gate",
        beam_width=beam_width, branch_limit=branch_limit, max_expansions=max_expansions,
        evaluated_expansions=expansions, completed_search_levels=levels,
        budget_exhausted=exhausted, greedy_completion_blocks=completion_count,
        full_route_reached_by_beam=not completion_count,
        prefix_progress=progress, track_count=engine.count, block_count=len(blocks),
        seeded_ties="global seeded input-block permutation; no title/identity musical columns",
        normalized_prefix_log_weight=float(best[1]), prefix_loss_scope=LOSS_SCOPE,
        prefix_normalization="allowed playable heads, or declared native-BPM bridge only when none are allowed; recomputed actual unplayed musical pool",
        forbidden_boundary_count=int(best[0]),
        preserved_locked_internal_forbidden=sum(not engine.allowed[engine.by_id[a], engine.by_id[b]]
            for block in blocks for a, b in zip(block, block[1:])),
        final_before_repair=final_loss, scoring_calls=engine.calls, context_cache_hits=engine.cache_hits,
        context_cache_limit=cache_limit, elapsed_seconds=time.perf_counter() - started,
        input_scope="BPM, key and remaining-pool connectivity; no future teacher labels, audio or sound-development arc",
        optimal_for_declared_objective=False, proprietary_equivalence=False,
        scope="bounded search, finite branch/beam caps and optional greedy budget completion; not a global optimum or audible quality claim")


def rescore_mode_order(records, profile, order, *, mode="fuzzy", priority="unknown", bpm_tolerance=None):
    """Recompute a repaired final route's diagnostic musical loss."""
    return DynamicModePool(records, profile, mode=mode, cache_limit=0,
        priority=priority, bpm_tolerance=bpm_tolerance).rescore_order(order)
