import asyncio
from collections import deque
from itertools import accumulate, chain, pairwise
from secrets import token_urlsafe
from typing import Iterable

import igraph as ig
from glhf.base import ClientProtocol
from glhf.bot import Bot
from glhf.client import SocketioClient
from glhf.gui import PygameGUI
from glhf.helper import patch
from glhf.server import make_2d_grid
from ortools.sat.python import cp_model as cp


def link_edges[T](edges: Iterable[tuple[T, T]], *, start: T = 0) -> deque[T]:
    es = deque(edges)
    if not es:
        return es  # type: ignore
    s, t = es.popleft()
    path = deque((s, t))
    while es:
        u, v = es.popleft()
        if t == u:
            t = v
            path.append(t)
        elif t == v:
            t = u
            path.append(t)
        elif s == u:
            s = v
            path.appendleft(s)
        elif s == v:
            s = u
            path.appendleft(s)
        else:
            es.append((u, v))
    if s != start:
        if t != start:
            raise ValueError("path does not start from the specified start vertex")
        path.reverse()
    return path


def scheduling_pcvrp(
    n_vertices: int,
    edges: list[tuple[int, int]],
    n_departs: int,
    moves_per_turn: int,
    turns_per_round: int,
    turns_per_sec: float,
    timeout: float,
    verbose: bool,
) -> list[tuple[int, deque[int]]]:
    model = cp.CpModel()
    new_int_var = model.new_int_var
    new_bool_var = model.new_bool_var
    add = model.add
    add_linear_constraint = model.add_linear_constraint
    add_exactly_one = model.add_exactly_one
    add_bool_or = model.add_bool_or
    add_implication = model.add_implication
    add_circuit = model.add_circuit
    add_hint = model.add_hint
    maximize = model.maximize
    Sum = cp.LinearExpr.Sum

    moves = turns_per_round * moves_per_turn  # 50
    ub = moves // (moves_per_turn + 1)  # 16

    lands_vars = [new_int_var(0, ub, f"#{i} lands") for i in range(n_departs)]
    time_vars = [new_int_var(0, ub, f"#{i} time") for i in range(n_departs)]

    add(lands_vars[0] * moves_per_turn + Sum(time_vars) == moves)
    sum_lands_var = Sum(lands_vars)
    add_linear_constraint(sum_lands_var, ub, turns_per_round - 1)  # 16 ~ 24

    for i in range(n_departs):
        add(lands_vars[i] <= time_vars[i])

    # lands[0] >= timeout * 2 / 1
    add(lands_vars[0] >= int(timeout * moves_per_turn / turns_per_sec))

    # lands[1] * 2 == time[0]
    add(lands_vars[1] * moves_per_turn == time_vars[0])

    # (lands[i] + lands[i + 1]) * 2 <= time[i - 1] + time[i]
    for i in range(1, n_departs - 1):
        add((lands_vars[i] + lands_vars[i + 1]) * 2 <= time_vars[i - 1] + time_vars[i])  # type: ignore

    for var, val in zip(lands_vars, (12, 6, 4, 2, 0)):
        add_hint(var, val)

    for var, val in zip(time_vars, (12, 8, 4, 2, 0)):
        add_hint(var, val)

    # heuristic
    for i, j in pairwise(range(n_departs)):
        add(lands_vars[i] >= lands_vars[j])
        add(time_vars[i] >= time_vars[j])

    visited_vars = [0] * (n_vertices - 1)
    edge_vars = []

    for i in range(n_departs):
        v_vars = [new_bool_var(f"#{i} v{v}") for v in range(1, n_vertices)]
        e_vars = [new_bool_var(f"#{i} e{e}") for e in edges]
        t_vars = [new_bool_var(f"#{i} t{t}") for t in range(n_vertices)]
        last_visited_vars = visited_vars
        visited_vars = [new_bool_var(f"#{i} visited_{v}") for v in range(1, n_vertices)]

        add(Sum(v_vars) <= time_vars[i])
        add(Sum(visited_vars) == Sum(last_visited_vars) + lands_vars[i])
        add_exactly_one(t_vars)

        for a, b, c in zip(v_vars, last_visited_vars, visited_vars):
            add_bool_or(a, b, c.Not())
            add_implication(a, c)
            add_implication(b, c)

        arcs = tuple(
            chain(
                ((v, v, var.Not()) for v, var in enumerate(v_vars, 1)),
                ((u, v, var) for (u, v), var in zip(edges, e_vars)),
                ((t, 0, var) for t, var in enumerate(t_vars)),
            )
        )
        add_circuit(arcs)

        edge_vars.append(e_vars)

    maximize(sum_lands_var)

    solver = cp.CpSolver()
    solver.parameters.max_time_in_seconds = timeout

    callback = (
        cp.VarArrayAndObjectiveSolutionPrinter(lands_vars + time_vars)
        if verbose
        else None
    )
    status = solver.Solve(model, callback)
    status_name = solver.StatusName()
    if status == cp.OPTIMAL or status == cp.FEASIBLE:
        if verbose:
            print(status_name)
        Value = solver.Value
        BooleanValue = solver.BooleanValue

        ts = accumulate(
            (Value(time_var) for time_var in time_vars),
            initial=Value(lands_vars[0] * 2),
        )
        paths = (
            link_edges(e for e, var in zip(edges, e_vars) if BooleanValue(var))
            for e_vars in edge_vars
        )

        return [(t, path) for t, path in zip(ts, paths)]

    raise ValueError(status_name)


def opening_moves(
    generals: list[int],
    graph: ig.Graph,
    player_index: int,
    *,
    n_departs: int = 5,
    d_max: int = 16,
    moves_per_turn: int = 2,
    turns_per_round: int = 25,
    turns_per_sec: float = 1.0,
    timeout: float = 5.0,
    verbose: bool = False,
) -> list[tuple[int, list[int]]]:
    if not (0 <= player_index < len(generals)):
        raise ValueError("player_index must be in [0, len(generals))")
    g_vs = graph.vs
    s = g_vs[generals[player_index]]["name"]
    g_vids = graph.neighborhood(s, d_max)
    h = graph.subgraph(g_vids)
    h_vids, layer, parents = h.bfs(s)
    n = len(h_vids)
    p = [0] * n
    for i, v in enumerate(h_vids):
        p[v] = i
    h = h.permute_vertices(p)
    h_edges = h.get_edgelist()
    path_list = scheduling_pcvrp(
        n,
        h_edges,
        n_departs,
        moves_per_turn,
        turns_per_round,
        turns_per_sec,
        timeout,
        verbose,
    )

    return [(time, [g_vids[v] for v in path]) for time, path in path_list]


class MyBot(Bot):
    async def run(self, client: ClientProtocol) -> None:
        queue_id = token_urlsafe(3)
        client.join_private(queue_id)

        async for data in self.queue_update:
            if not data["isForcing"]:
                client.set_force_start(True)

        await self.game_start.wait()

        map_: list[int] = []
        cities: list[int] = []

        paths: list[tuple[int, list[int]]] = []

        async for data in self.game_update:
            map_ = patch(map_, data["map_diff"])
            cities = patch(cities, data["cities_diff"])
            generals = data["generals"]
            turn = data["turn"]
            if turn == 1:
                col = map_[0]
                row = map_[1]
                size = col * row
                graph = make_2d_grid(row, col)
                terrain = map_[2 + size :]
                obstacles = [i for i, t in enumerate(terrain) if t == -2 or t == -4]
                obstacles += cities
                graph.delete_edges(_source=obstacles)

                coro = asyncio.to_thread(opening_moves, generals, graph, 0)
                task = asyncio.create_task(coro)
                task.add_done_callback(lambda t: paths.extend(t.result()))

            elif turn == 50:
                await client.surrender()

            if paths:
                s, path = paths[0]
                t = s + len(path) - 1
                if s <= turn < t:
                    i = turn - s
                    await client.attack(path[i], path[i + 1], False)
                    if turn == t - 1:
                        paths.pop(0)

        await client.leave_game()


if __name__ == "__main__":
    USERID = "123"
    USERNAME = "[BOT] 123"

    bot = MyBot()
    gui = PygameGUI()

    client = SocketioClient(
        USERID,
        USERNAME,
        bot,
        gui,
    )

    asyncio.run(client.run(), debug=False)
