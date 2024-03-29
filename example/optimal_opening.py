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


def link_edges(edges: Iterable[tuple[int, int]], *, start: int = 0) -> list[int]:
    # n = len(edges), time = O(n ^ 2)
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


class SchedulingCallback(cp.CpSolverSolutionCallback):
    def __init__(
        self,
        n_lands_vars: list[cp.IntVar],
        n_turns_vars: list[cp.IntVar],
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


def scheduling_pcvrp(
    n_vertices: int,
    edges: list[tuple[int, int]],
    n_paths: int = 5,
    mpt: int = 2,
    tpr: int = 25,
    tps: float = 1.0,
    timeout: float = 3.0,
    verbose: bool = False,
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

    # hints
    # n_lands = [12, 6, 4, 2, 0]
    # n_turns = [12, 8, 4, 2, 0]
    for var, n_lands in zip(n_lands_vars, (12, 6, 4, 2, 0)):
        model.add_hint(var, n_lands)
    for var, start_turns in zip(n_turns_vars, (12, 8, 4, 2, 0)):
        model.add_hint(var, start_turns)

    # heuristic
    # n_lands[i] >= n_lands[i+1] for i in range(5-1)
    # n_turns[i] >= n_turns[i+1] for i in range(5-1)
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
            model.new_bool_var(f"#{i} l{v}") for v in range(1, n_vertices)
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
