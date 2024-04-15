import time
from collections import deque
from itertools import accumulate, chain, pairwise
from typing import Iterable, Sequence

import igraph as ig
from glhf.server._local import make_2d_grid
from ortools.sat.python import cp_model as cp


class SchedulingCallback(cp.CpSolverSolutionCallback):
    def __init__(
        self,
        n_lands_vars: Sequence[cp.IntVar],
        n_turns_vars: Sequence[cp.IntVar],
    ) -> None:
        super().__init__()
        self.n_lands_vars = n_lands_vars
        self.n_turns_vars = n_turns_vars
        self.count = 0

    def on_solution_callback(self) -> None:
        self.count += 1
        print(
            f"solution {self.count}, obj = {int(self.objective_value)}, time = {self.wall_time:.2f}s",
            f"lands [{", ".join(f"{self.value(var):>2}" for var in self.n_lands_vars)}]",
            f"turns [{", ".join(f"{self.value(var):>2}" for var in self.n_turns_vars)}]",
            sep="\n",
            end="\n\n",
        )


def make_graph(map_: list[int], cities: list[int], generals: list[int]) -> ig.Graph:
    col = map_[0]
    row = map_[1]
    size = col * row
    terrain = map_[2 + size :]
    graph = make_2d_grid(row, col)
    # vs = graph.vs
    # vs[cities]["is_city"] = True
    # vs[generals]["is_general"] = True
    # vs["army"] = map_[2 : 2 + size]
    # vs["terrain"] = map_[2 + size :]
    obstacles = [i for i, t in enumerate(terrain) if t == -2 or t == -4]
    obstacles += cities
    graph.delete_edges(_source=obstacles)
    return graph


def link_edges(edges: Iterable[tuple[int, int]], *, start: int = 0) -> list[int]:
    # time = O(len(edges) ^ 2)
    q = deque(edges)
    path = [start]
    while q:
        u, v = q.popleft()
        if path[-1] == u:
            path.append(v)
        elif path[-1] == v:
            path.append(u)
        else:
            q.append((v, u))
    if path[-1] == start:
        path.reverse()
    if path[0] == start:
        return path
    raise ValueError("path does not start from the specified start vertex")


# TODO fix remainder bug in scheduling
# for example, now we have:
# .  | 5 | 3 | 1
# 11 | 6 | 3 | .
# but it supposed to be:
# .  | 5 | 3 | 2
# 11 | 6 | 3 | .
def scheduling_pcvrp(
    n_vertices: int,
    edges: Sequence[tuple[int, int]],
    n_paths: int,
    n_lands_hints: Sequence[int] | None,
    n_turns_hints: Sequence[int] | None,
    mpt: int,
    tpr: int,
    tps: float,
    timeout: float,
    heuristics: bool,
    verbose: bool,
) -> list[tuple[int, list[int]]]:
    t_start = time.monotonic()
    solver = cp.CpSolver()
    model = cp.CpModel()

    n_moves = tpr * mpt  # 50
    max_tpr = n_moves // (mpt + 1)  # 16
    min_start_lands = int(timeout * mpt / tps)  # 6

    n_lands_vars = [
        model.new_int_var(0, max_tpr, f"#{i} lands") for i in range(n_paths)
    ]
    n_turns_vars = [
        model.new_int_var(0, max_tpr, f"#{i} turns") for i in range(n_paths)
    ]
    sum_lands_var = cp.LinearExpr.sum(n_lands_vars)
    sum_turns_var = cp.LinearExpr.sum(n_turns_vars)
    start_turns_var = n_lands_vars[0] * mpt

    # n_lands[0] * 2 + sum(n_turns) == n_moves
    model.add(start_turns_var + sum_turns_var == n_moves)

    # 16 <= sum(n_lands) <= 24
    model.add_linear_constraint(sum_lands_var, max_tpr, tpr - 1)

    # (n_lands[i] <= n_turns[i]) for i in range(5)
    for i in range(n_paths):
        model.add(n_lands_vars[i] <= n_turns_vars[i])

    # n_lands[0] >= timeout * 2 / 1
    model.add(n_lands_vars[0] >= min_start_lands)

    # n_lands[1] * 2 == n_turns[0]
    model.add(n_lands_vars[1] * mpt == n_turns_vars[0])

    # n_lands[i] + n_lands[i + 1] <= (n_turns[i - 1] + n_turns[i]) / 2
    for i in range(1, n_paths - 1):
        n_lands_mul_mpt_var = (n_lands_vars[i] + n_lands_vars[i + 1]) * mpt
        n_turns_var = n_turns_vars[i - 1] + n_turns_vars[i]
        model.add(n_lands_mul_mpt_var <= n_turns_var)  # type: ignore

    # n_lands_hints = [12, 6, 4, 2, 0]
    if n_lands_hints is not None:
        assert len(n_lands_hints) == n_paths, "invalid length of `n_lands_hints`"
        for var, n_lands in zip(n_lands_vars, n_lands_hints):
            model.add_hint(var, n_lands)

    # n_turns_hints = [12, 8, 4, 2, 0]
    if n_turns_hints is not None:
        assert len(n_turns_hints) == n_paths, "invalid length of `n_turns_hints`"

        for var, n_turns in zip(n_turns_vars, n_turns_hints):
            model.add_hint(var, n_turns)

    # heuristic
    # n_lands[i] >= n_lands[i+1] for i in range(5 - 1)
    # n_turns[i] >= n_turns[i+1] for i in range(5 - 1)
    if heuristics:
        for i, j in pairwise(range(n_paths)):
            model.add(n_lands_vars[i] >= n_lands_vars[j])
            model.add(n_turns_vars[i] >= n_turns_vars[j])

    # objective
    model.maximize(sum_lands_var)

    edge_vars_list = []
    new_visited_vars = [0] * (n_vertices - 1)

    for i in range(n_paths):
        vertex_vars = [
            model.new_bool_var(f"#{i} vertex-{v}") for v in range(1, n_vertices)
        ]
        edge_vars = [model.new_bool_var(f"#{i} edge-{u}-{v}") for u, v in edges]
        sink_vars = [model.new_bool_var(f"#{i} sink-{t}") for t in range(n_vertices)]

        old_visited_vars = new_visited_vars
        new_visited_vars = [
            model.new_bool_var(f"#{i} visited-{v}") for v in range(1, n_vertices)
        ]

        sum_vertex_var = cp.LinearExpr.sum(vertex_vars)
        sum_old_visited_var = cp.LinearExpr.sum(old_visited_vars)
        sum_new_visited_var = cp.LinearExpr.sum(new_visited_vars)

        model.add(sum_vertex_var <= n_turns_vars[i])
        model.add(sum_new_visited_var == sum_old_visited_var + n_lands_vars[i])
        model.add_exactly_one(sink_vars)

        for a, b, c in zip(vertex_vars, old_visited_vars, new_visited_vars):
            model.add_bool_or(a, b, c.negated())
            model.add_implication(a, c)
            model.add_implication(b, c)

        arcs = tuple(
            chain(
                ((v, v, var.negated()) for v, var in enumerate(vertex_vars, 1)),
                ((u, v, var) for (u, v), var in zip(edges, edge_vars)),
                ((t, 0, var) for t, var in enumerate(sink_vars)),
            )
        )
        model.add_circuit(arcs)

        edge_vars_list.append(edge_vars)

    # solve
    callback = SchedulingCallback(n_lands_vars, n_turns_vars) if verbose else None
    t_end = time.monotonic()
    solver.parameters.max_time_in_seconds = timeout - (t_end - t_start)
    status = solver.solve(model, callback)

    status_name = solver.status_name()
    if verbose:
        print(status_name)

    if status in (cp.OPTIMAL, cp.FEASIBLE):
        start_turns = accumulate(
            (solver.value(var) for var in n_turns_vars),
            initial=solver.value(start_turns_var),
        )
        paths = (
            link_edges(e for e, var in zip(edges, vars) if solver.boolean_value(var))
            for vars in edge_vars_list
        )
        return list(zip(start_turns, paths))
    else:
        if status == cp.MODEL_INVALID:
            print(model.validate())
        raise ValueError(status_name)


def opening_moves(
    graph: ig.Graph,
    general_index: int,
    *,
    n_paths: int = 5,
    d_max: int = 16,
    mpt: int = 2,
    tpr: int = 25,
    tps: float = 1.0,
    timeout: float = 3.0,
    heuristics: bool = True,
    verbose: bool = False,
) -> list[tuple[int, list[int]]]:
    g_vs = graph.vs
    s_name = g_vs[general_index]["name"]
    g_vids = graph.neighborhood(s_name, d_max)

    h = graph.subgraph(g_vids)
    h_vids, layer, parents = h.bfs(s_name)
    n = len(h_vids)

    inv = [0] * n
    for i, v in enumerate(h_vids):
        inv[v] = i

    h = h.permute_vertices(inv)
    h_edges = h.get_edgelist()

    n_lands_hints = 12, 6, 4, 2, 0
    n_turns_hints = 12, 8, 4, 2, 0

    h_paths = scheduling_pcvrp(
        n,
        h_edges,
        n_paths,
        n_lands_hints,
        n_turns_hints,
        mpt,
        tpr,
        tps,
        timeout,
        heuristics,
        verbose,
    )
    return [(turn, [g_vids[v] for v in path]) for turn, path in h_paths]
