import asyncio
import time
from collections import deque
from itertools import accumulate, chain, pairwise
from secrets import token_urlsafe
from typing import Iterable

import igraph as ig
from glhf.base import ClientProtocol
from glhf.bot import Bot
from glhf.client import BasicClient
from glhf.gui import PygameGUI
from glhf.helper import patch
from glhf.server import LocalServer, make_2d_grid
from ortools.sat.python import cp_model as cp


def link_edges[T](edges: Iterable[tuple[T, T]], *, start: T = 0) -> deque[T]:
    es = deque(edges)
    path = deque()
    if not es:
        return path
    s, t = es.popleft()
    path += s, t
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


class SchedulingCallback(cp.CpSolverSolutionCallback):
    def __init__(self, lands_vars: list[cp.IntVar], time_vars: list[cp.IntVar]) -> None:
        super().__init__()
        self.lands_vars = lands_vars
        self.time_vars = time_vars
        self.count: int = 0

    def on_solution_callback(self) -> None:
        self.count += 1
        print(
            f"solution {self.count}, obj = {int(self.objective_value)}, time = {self.wall_time:.2f}s",
            f"lands [{", ".join(f"{self.value(v):>2}" for v in self.lands_vars)}]",
            f"time  [{", ".join(f"{self.value(v):>2}" for v in self.time_vars)}]",
            sep="\n",
            end="\n\n",
        )


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
    t_start = time.monotonic()

    m = cp.CpModel()
    cp_sum = cp.LinearExpr.sum

    moves = turns_per_round * moves_per_turn  # 50
    ub = moves // (moves_per_turn + 1)  # 16

    lands_vars = [m.new_int_var(0, ub, f"#{i} lands") for i in range(n_departs)]
    time_vars = [m.new_int_var(0, ub, f"#{i} time") for i in range(n_departs)]

    m.add(lands_vars[0] * moves_per_turn + cp_sum(time_vars) == moves)
    sum_lands_var = cp_sum(lands_vars)
    m.add_linear_constraint(sum_lands_var, ub, turns_per_round - 1)  # 16 ~ 24

    for i in range(n_departs):
        m.add(lands_vars[i] <= time_vars[i])

    # lands[0] >= timeout * 2 / 1
    m.add(lands_vars[0] >= int(timeout * moves_per_turn / turns_per_sec))

    # lands[1] * 2 == time[0]
    m.add(lands_vars[1] * moves_per_turn == time_vars[0])

    # (lands[i] + lands[i + 1]) * 2 <= time[i - 1] + time[i]
    for i in range(1, n_departs - 1):
        m.add(
            (lands_vars[i] + lands_vars[i + 1]) * 2 <= time_vars[i - 1] + time_vars[i]  # type: ignore
        )

    # hints
    for var, val in zip(lands_vars, (12, 6, 4, 2, 0)):
        m.add_hint(var, val)

    for var, val in zip(time_vars, (12, 8, 4, 2, 0)):
        m.add_hint(var, val)

    # heuristic
    for i, j in pairwise(range(n_departs)):
        m.add(lands_vars[i] >= lands_vars[j])
        m.add(time_vars[i] >= time_vars[j])

    visited_vars = [0] * (n_vertices - 1)
    edge_vars = []

    for i in range(n_departs):
        v_vars = [m.new_bool_var(f"#{i} v{v}") for v in range(1, n_vertices)]
        e_vars = [m.new_bool_var(f"#{i} e{e}") for e in edges]
        t_vars = [m.new_bool_var(f"#{i} t{t}") for t in range(n_vertices)]
        last_visited_vars = visited_vars
        visited_vars = [m.new_bool_var(f"#{i} ^{v}") for v in range(1, n_vertices)]

        m.add(cp_sum(v_vars) <= time_vars[i])
        m.add(cp_sum(visited_vars) == cp_sum(last_visited_vars) + lands_vars[i])
        m.add_exactly_one(t_vars)

        for a, b, c in zip(v_vars, last_visited_vars, visited_vars):
            m.add_bool_or(a, b, c.Not())
            m.add_implication(a, c)
            m.add_implication(b, c)

        arcs = tuple(
            chain(
                ((v, v, var.negated()) for v, var in enumerate(v_vars, 1)),
                ((u, v, var) for (u, v), var in zip(edges, e_vars)),
                ((t, 0, var) for t, var in enumerate(t_vars)),
            )
        )
        m.add_circuit(arcs)

        edge_vars.append(e_vars)

    m.maximize(sum_lands_var)

    solver = cp.CpSolver()

    callback = SchedulingCallback(lands_vars, time_vars) if verbose else None

    t_end = time.monotonic()
    solver.parameters.max_time_in_seconds = timeout - (t_end - t_start)

    status = solver.solve(m, callback)
    status_name = solver.status_name()

    if status in (cp.OPTIMAL, cp.FEASIBLE):
        if verbose:
            print(status_name)

        ts = accumulate(
            (solver.value(time_var) for time_var in time_vars),
            initial=solver.value(lands_vars[0] * 2),
        )
        paths = (
            link_edges(e for e, var in zip(edges, e_vars) if solver.boolean_value(var))
            for e_vars in edge_vars
        )

        return [(t, path) for t, path in zip(ts, paths)]

    if verbose:
        print(status_name, m.validate(), sep="\n")

    raise ValueError(status_name)


def opening_moves(
    graph: ig.Graph,
    general_index: int,
    *,
    n_departs: int = 5,
    d_max: int = 16,
    moves_per_turn: int = 2,
    turns_per_round: int = 25,
    turns_per_sec: float = 1.0,
    timeout: float = 5.0,
    verbose: bool = False,
) -> list[tuple[int, list[int]]]:
    vs = graph.vs
    s = vs[general_index]["name"]
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


class OptimalOpening(Bot):
    async def run(self, client: ClientProtocol) -> None:
        client.join_private("" or token_urlsafe(3))

        async for data in self.queue_update:
            if not data["isForcing"]:
                client.set_force_start(True)

        data = await self.game_start.wait()
        assert data is not None, "stream is closed"
        player_index = data["playerIndex"]

        map_: list[int] = []
        cities: list[int] = []
        paths: list[tuple[int, list[int]]] = []
        graph = ig.Graph()  # avoid unbound

        async for data in self.game_update:
            turn = data["turn"]
            map_ = patch(map_, data["map_diff"])
            cities = patch(cities, data["cities_diff"])
            generals = data["generals"]

            if turn == 1:
                graph = make_graph(map_, cities, generals)
                task = asyncio.create_task(
                    asyncio.to_thread(
                        opening_moves,
                        graph,
                        generals[player_index],
                        timeout=3.0,
                        verbose=True,
                    )
                )
                task.add_done_callback(
                    lambda t: paths.extend(t.result())
                    if t.exception() is None
                    else t.print_stack()
                )

            elif turn == 50:
                client.surrender()

            if paths:
                s, path = paths[0]
                t = s + len(path) - 1
                if s <= turn < t:
                    i = turn - s
                    client.attack(path[i], path[i + 1], False)
                    if turn == t - 1:
                        paths.pop(0)

        client.leave_game()


def set_eager_task_factory(is_eager: bool) -> None:
    loop = asyncio.get_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory if is_eager else None)  # type: ignore


async def main() -> None:
    set_eager_task_factory(True)
    USERID = "h4K1gOyHNnkGngym8fUuYA"
    USERNAME = "PsittaTestBot"
    server = LocalServer(18, 16)
    # server = SocketioServer()
    bot = OptimalOpening()
    gui = PygameGUI()
    client = BasicClient(USERID, USERNAME, bot, gui, server)
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
