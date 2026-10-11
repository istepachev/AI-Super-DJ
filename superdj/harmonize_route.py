"""Bounded, auditable directed whole-path search for sequencing groups.

The caller supplies its own nonnegative objective matrix. This module knows
nothing about music keys or proprietary ordering rules. Every vertex must
occur exactly once in an open Hamiltonian path; fixed endpoints are optional.
"""
from __future__ import annotations

import math
from numbers import Integral
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


MAX_NODES = 100
MAX_TIME_LIMIT = 120.0
_INTEGRAL_TOLERANCE = 1e-5


def _matrix(costs) -> np.ndarray:
    try:
        original = np.asarray(costs)
    except (TypeError, ValueError) as exc:
        raise ValueError("costs must be a square numeric matrix") from exc
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("costs must be a square numeric matrix")
    if np.iscomplexobj(original) or original.dtype.kind in {"b", "U", "S"}:
        raise ValueError("costs must contain real nonnegative numbers")
    n = original.shape[0]
    if not 1 <= n <= MAX_NODES:
        raise ValueError(f"costs requires 1 to {MAX_NODES} vertices")
    # Inspect original values as well: converting booleans or strings into
    # floats could conceal a malformed objective supplied by the caller.
    for row in costs:
        for value in row:
            if isinstance(value, (bool, np.bool_, str, bytes)):
                raise ValueError("costs must contain finite nonnegative numbers")
    try:
        result = np.asarray(costs, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("costs must contain finite nonnegative numbers") from exc
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("costs must contain finite nonnegative numbers")
    if not math.isfinite(float(np.max(result)) * max(1, n - 1)):
        raise ValueError("The complete path objective must remain finite")
    return result


def _endpoint(value, n, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < n:
        raise ValueError(f"{label} must be a vertex index in 0..{n - 1}")
    return int(value)


def _finite_optional(value):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def solve_hamiltonian_path(costs, *, first=None, last=None, time_limit=20.0):
    """Return a verified incumbent, or ``None`` when the solver found none.

    Binary directed edges, one start and one end encode an open path.
    Continuous MTZ positions prohibit detached cycles. A time-limit incumbent
    is useful but is not labelled optimal. Feasibility and cost are rechecked
    independently of the optimizer's status before returning any result.
    """
    matrix = _matrix(costs)
    n = len(matrix)
    first = _endpoint(first, n, "first")
    last = _endpoint(last, n, "last")
    if first is not None and first == last and n > 1:
        raise ValueError("A multi-vertex path requires distinct endpoints")
    if isinstance(time_limit, (bool, np.bool_, str, bytes)):
        raise ValueError("time_limit must be a finite positive number at most 120")
    seconds = _finite_optional(time_limit)
    if seconds is None or not 0 < seconds <= MAX_TIME_LIMIT:
        raise ValueError("time_limit must be a finite positive number at most 120")
    if n == 1:
        return {"order": [0], "cost": 0.0, "optimal_for_objective": True,
                "mip_status": 0, "mip_message": "Single-vertex open path",
                "mip_gap": 0.0, "mip_node_count": 0,
                "method": "directed_hamiltonian_path_milp"}

    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    edge_count = len(edges)
    start_offset, end_offset, position_offset = edge_count, edge_count + n, edge_count + 2 * n
    variable_count = edge_count + 3 * n
    objective = np.zeros(variable_count)
    objective[:edge_count] = [matrix[i, j] for i, j in edges]
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:position_offset] = 1
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    upper[position_offset:] = n - 1
    for index, offset in ((first, start_offset), (last, end_offset)):
        if index is not None:
            lower[offset + index] = upper[offset + index] = 1

    rows, columns, coefficients, constraint_lower, constraint_upper = [], [], [], [], []

    def constraint(entries, low, high):
        row = len(constraint_lower)
        for column, coefficient in entries:
            rows.append(row)
            columns.append(column)
            coefficients.append(coefficient)
        constraint_lower.append(low)
        constraint_upper.append(high)

    outgoing, incoming = [[] for _ in range(n)], [[] for _ in range(n)]
    for edge_index, (i, j) in enumerate(edges):
        outgoing[i].append((edge_index, 1))
        incoming[j].append((edge_index, 1))
    for vertex in range(n):
        constraint([*outgoing[vertex], (end_offset + vertex, 1)], 1, 1)
        constraint([*incoming[vertex], (start_offset + vertex, 1)], 1, 1)
    constraint([(start_offset + vertex, 1) for vertex in range(n)], 1, 1)
    constraint([(end_offset + vertex, 1) for vertex in range(n)], 1, 1)
    for edge_index, (i, j) in enumerate(edges):
        # Active edge i->j implies u[j] >= u[i]+1. Summing this
        # inequality around any cycle gives an immediate contradiction.
        constraint([(position_offset + i, 1), (position_offset + j, -1),
                    (edge_index, n)], -np.inf, n - 1)
    sparse = coo_array((np.asarray(coefficients, dtype=float),
                        (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32))),
                       shape=(len(constraint_lower), variable_count)).tocsc()
    try:
        result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(sparse, constraint_lower, constraint_upper),
                      options={"time_limit": seconds, "mip_rel_gap": 0.0, "disp": False})
    except RuntimeError:
        return None
    raw = getattr(result, "x", None)
    if raw is None:
        return None
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if values.shape != (variable_count,) or not np.isfinite(values).all():
        return None
    rounded = np.rint(values[:position_offset])
    if (np.abs(values[:position_offset] - rounded) > _INTEGRAL_TOLERANCE).any():
        return None
    if ((rounded < 0) | (rounded > 1)).any():
        return None
    starts = np.flatnonzero(rounded[start_offset:end_offset])
    ends = np.flatnonzero(rounded[end_offset:position_offset])
    if len(starts) != 1 or len(ends) != 1:
        return None
    selected = [pair for index, pair in enumerate(edges) if rounded[index] == 1]
    if len(selected) != n - 1:
        return None
    successor, predecessor = {}, {}
    for i, j in selected:
        if i in successor or j in predecessor:
            return None
        successor[i], predecessor[j] = j, i
    start, end = int(starts[0]), int(ends[0])
    if start in predecessor or end in successor:
        return None
    order, seen = [start], {start}
    while order[-1] in successor:
        next_vertex = successor[order[-1]]
        if next_vertex in seen:
            return None
        seen.add(next_vertex)
        order.append(next_vertex)
    if len(order) != n or order[-1] != end or set(order) != set(range(n)):
        return None
    if (first is not None and order[0] != first) or (last is not None and order[-1] != last):
        return None
    cost = math.fsum(float(matrix[i, j]) for i, j in zip(order, order[1:]))
    reported_cost = _finite_optional(getattr(result, "fun", None))
    if reported_cost is not None and not math.isclose(cost, reported_cost, rel_tol=1e-7, abs_tol=1e-6):
        return None
    status = getattr(result, "status", None)
    status = int(status) if isinstance(status, Integral) and not isinstance(status, bool) else None
    node_count = _finite_optional(getattr(result, "mip_node_count", None))
    return {"order": order, "cost": cost,
            "optimal_for_objective": status == 0 and reported_cost is not None,
            "mip_status": status, "mip_message": str(getattr(result, "message", "")),
            "mip_gap": _finite_optional(getattr(result, "mip_gap", None)),
            "mip_node_count": int(node_count) if node_count is not None else None,
            "method": "directed_hamiltonian_path_milp"}
