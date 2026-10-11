"""Independent feasibility and optimum checks for directed whole-path search."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from superdj import harmonize_route as route


def exhaustive(matrix, first=None, last=None):
    n = len(matrix)
    candidates = (order for order in itertools.permutations(range(n))
                  if (first is None or order[0] == first)
                  and (last is None or order[-1] == last))
    return min((sum(matrix[a][b] for a, b in zip(order, order[1:])), order)
               for order in candidates)


@pytest.mark.parametrize("first,last", [(None, None), (0, None), (None, 4), (0, 4), (3, 1)])
def test_matches_exhaustive_directed_optimum_with_endpoints(first, last):
    matrix = [[0, 8, 4, 9, 3], [2, 0, 7, 3, 11], [6, 1, 0, 8, 6],
              [4, 7, 2, 0, 5], [9, 3, 6, 1, 0]]
    minimum, _ = exhaustive(matrix, first, last)
    result = route.solve_hamiltonian_path(matrix, first=first, last=last, time_limit=5)
    assert result is not None
    assert result["cost"] == pytest.approx(minimum)
    assert result["optimal_for_objective"]
    assert len(result["order"]) == len(set(result["order"])) == len(matrix)
    assert first is None or result["order"][0] == first
    assert last is None or result["order"][-1] == last
    assert sum(matrix[a][b] for a, b in zip(result["order"], result["order"][1:])) == result["cost"]


def test_preserves_direction_instead_of_symmetrizing_cheap_reverse_edges():
    matrix = [[0, 1, 50, 50], [50, 0, 1, 50], [50, 50, 0, 1], [50, 50, 50, 0]]
    result = route.solve_hamiltonian_path(matrix, first=3, last=0, time_limit=5)
    assert result["cost"] == 101
    assert result["order"][0] == 3 and result["order"][-1] == 0
    reverse = route.solve_hamiltonian_path(matrix, first=0, last=3, time_limit=5)
    assert reverse["order"] == [0, 1, 2, 3]
    assert reverse["cost"] == 3


def test_cannot_choose_detached_subcycle_even_when_edges_are_cheaper():
    matrix = [[0, 1, 90, 90], [1, 0, 90, 90], [90, 90, 0, 1], [90, 90, 1, 0]]
    result = route.solve_hamiltonian_path(matrix, time_limit=5)
    assert result is not None
    assert result["cost"] == 92
    assert set(result["order"]) == {0, 1, 2, 3}


def test_single_vertex_has_no_self_edge_cost_and_needs_no_optimizer(monkeypatch):
    monkeypatch.setattr(route, "milp", lambda *a, **k: pytest.fail("unneeded MILP call"))
    result = route.solve_hamiltonian_path([[20]], first=0, last=0)
    assert result["order"] == [0]
    assert result["cost"] == 0
    assert result["optimal_for_objective"]


@pytest.mark.parametrize("matrix", [[], [1, 2], [[1, 2]], [[0, 1], [2]],
                                     [[0, -1], [2, 0]], [[0, float("inf")], [1, 0]],
                                     [[0, float("nan")], [1, 0]], [[0, True], [1, 0]],
                                     [[0, "1"], [2, 0]], [[0, 1j], [2, 0]], np.zeros((101, 101))])
def test_rejects_invalid_or_unbounded_matrix(matrix):
    with pytest.raises(ValueError):
        route.solve_hamiltonian_path(matrix)


@pytest.mark.parametrize("kwargs", [dict(first=2), dict(last=-1), dict(first=True),
                                     dict(last=1.5), dict(first=0, last=0),
                                     dict(time_limit=0), dict(time_limit=121),
                                     dict(time_limit=float("inf")), dict(time_limit=float("nan")),
                                     dict(time_limit=True), dict(time_limit="2")])
def test_rejects_invalid_endpoints_and_time_budget(kwargs):
    with pytest.raises(ValueError):
        route.solve_hamiltonian_path([[0, 1], [2, 0]], **kwargs)


def incumbent(matrix, path, *, status=1):
    n = len(matrix)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    values = np.zeros(len(edges) + 3 * n)
    selected = set(zip(path, path[1:]))
    for index, edge in enumerate(edges):
        values[index] = edge in selected
    values[len(edges) + path[0]] = 1
    values[len(edges) + n + path[-1]] = 1
    for position, vertex in enumerate(path):
        values[len(edges) + 2 * n + vertex] = position
    return SimpleNamespace(x=values, status=status, message="Time limit", mip_gap=.4,
                           fun=sum(matrix[a][b] for a, b in zip(path, path[1:])), mip_node_count=12)


def test_time_limit_keeps_verified_incumbent_without_optimal_claim(monkeypatch):
    matrix = [[0, 2, 9], [4, 0, 3], [2, 1, 0]]
    fake = incumbent(matrix, [0, 1, 2])
    monkeypatch.setattr(route, "milp", lambda *a, **k: fake)
    result = route.solve_hamiltonian_path(matrix, first=0, last=2, time_limit=.01)
    assert result["order"] == [0, 1, 2]
    assert result["cost"] == 5
    assert not result["optimal_for_objective"]
    assert result["mip_status"] == 1 and result["mip_gap"] == .4


@pytest.mark.parametrize("raw", [None, np.array([]), np.array([float("nan")])])
def test_missing_or_malformed_incumbent_returns_none(monkeypatch, raw):
    monkeypatch.setattr(route, "milp", lambda *a, **k: SimpleNamespace(x=raw, status=1))
    assert route.solve_hamiltonian_path([[0, 1], [1, 0]], time_limit=.01) is None


def test_detects_disconnected_cycle_in_returned_incumbent(monkeypatch):
    matrix = np.ones((4, 4))
    fake = incumbent(matrix, [0, 1])
    edges = [(i, j) for i in range(4) for j in range(4) if i != j]
    fake.x[edges.index((2, 3))] = fake.x[edges.index((3, 2))] = 1
    fake.fun = 3
    monkeypatch.setattr(route, "milp", lambda *a, **k: fake)
    assert route.solve_hamiltonian_path(matrix) is None


@pytest.mark.parametrize("failure", ["fractional", "wrong_cost", "wrong_endpoint"])
def test_verifies_binary_edges_cost_and_locked_endpoint(monkeypatch, failure):
    matrix = [[0, 1, 2], [2, 0, 3], [1, 2, 0]]
    fake = incumbent(matrix, [0, 1, 2])
    if failure == "fractional":
        fake.x[0] = .5
    elif failure == "wrong_cost":
        fake.fun = 99
    monkeypatch.setattr(route, "milp", lambda *a, **k: fake)
    kwargs = {"first": 1} if failure == "wrong_endpoint" else {}
    assert route.solve_hamiltonian_path(matrix, **kwargs) is None


def test_optimizer_failure_without_incumbent_is_not_fabricated(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(route, "milp", unavailable)
    assert route.solve_hamiltonian_path([[0, 1], [1, 0]]) is None
