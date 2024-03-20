import asyncio
import time
from collections import deque
from itertools import combinations as C
from itertools import product as P
from random import Random
from typing import Any

import igraph as ig
import ortools.sat.python.cp_model as cp
from socketio import AsyncClient

from glhf.base import ClientProtocol, ServerProtocol
from glhf.helper import make_diff
from glhf.utils import to_task

WSURL = "wss://ws.generals.io"
BOTKEY = "sd09fjd203i0ejwi_changeme"


def make_2d_grid(
    row: int,
    col: int,
) -> ig.Graph:
    """Creates a 2D grid graph.

    Args:
        row: The number of rows in the map.
        col: The number of columns in the map.

    Returns:
        graph: The 2D grid graph.
    """
    g = ig.Graph.Lattice((col, row), circular=False)
    coords = [(x, y) for y, x in P(range(row), range(col))]
    vs = g.vs
    vs["coord"] = coords
    vs["name"] = [f"{x},{y}" for x, y in coords]
    vs["army"] = 0
    vs["terrain"] = -1
    return g


def make_complete_map(
    row: int,
    col: int,
) -> tuple[list[int], ig.Graph]:
    """Creates a complete map and its corresponding graph.

    Args:
        row: The number of rows in the map.
        col: The number of columns in the map.

    Returns:
        A tuple containing the map and the graph.
    """
    g = ig.Graph.Lattice((col, row), circular=False)
    g["row"] = row
    g["col"] = col
    coords = [(x, y) for y, x in P(range(row), range(col))]
    vs = g.vs
    vs["coord"] = coords
    vs["name"] = [f"{x},{y}" for x, y in coords]
    n = col * row
    m = [col, row, *(0,) * n, *(-1,) * n]
    return m, g


def p_dispersion[T](
    p: int,
    d_min: int,
    d_max: int,
    selects: list[T],
    coords: list[tuple[int, int]],
    verbose: bool,
) -> list[T]:
    """Finds a dispersion of points on a grid.

    Args:
        p: The number of points to select.
        d_min: The minimum distance between any two selected points.
        d_max: The maximum distance between any two selected points.
        selects: The list of selectable items.
        coords: The coordinates of the selectable items.
        callback: An optional callback function for the solver.

    Returns:
        A list of selected items.

    Raises:
        ValueError: If the solver fails to find a solution.
    """
    model = cp.CpModel()
    new_int_var = model.new_int_var
    new_bool_var = model.new_bool_var
    add = model.add
    Sum = cp.LinearExpr.Sum

    d_min_var = new_int_var(d_min, d_max, "d_min")
    select_vars = [new_bool_var(str(c)) for c in coords]
    add(Sum(select_vars) == p)
    for (i, j), ((x1, y1), (x2, y2)) in zip(C(select_vars, 2), C(coords, 2)):
        d = abs(x1 - x2) + abs(y1 - y2)
        add((2 - i - j) * d_min + d >= d_min_var)  # type: ignore

    solver = cp.CpSolver()
    callback = cp.VarArraySolutionPrinter((d_min_var,)) if verbose else None

    if solver.solve(model, callback) == cp.OPTIMAL:
        boolean_value = solver.boolean_value
        return [i for i, v in zip(selects, select_vars) if boolean_value(v)]

    raise ValueError(solver.status_name())


def move_army(
    graph: ig.Graph,
    player_id: int,
    start: int,
    end: int,
    is50: bool,
) -> bool:
    if graph.are_connected(start, end):
        vs = graph.vs
        u = vs[start]
        v = vs[end]
        u_army = u["army"]
        v_army = v["army"]
        u_terrain = u["terrain"]
        v_terrain = v["terrain"]
        if u_army > 1 and u_terrain == player_id and v_terrain >= -1:
            if is50:
                army = u_army // 2
                u["army"] = army
                army = u_army - army
            else:
                u["army"] = 1
                army = u_army - 1
            if u_terrain == v_terrain:
                v["army"] += army
            else:
                if v_army >= army:
                    v["army"] -= army
                else:
                    v["army"] = army - v_army
                    v["terrain"] = u_terrain
            return True
    return False


class LocalServer(ServerProtocol):
    def __init__(self, row: int, col: int) -> None:
        self.clients: dict[ClientProtocol, str] = {}
        self.row = row
        self.col = col
        self.map_ = []
        self.graph = make_2d_grid(row, col)
        self.turn = 0
        self.generals: list[int] = []
        self.cities: list[int] = []
        self.usernames: dict[str, str] = {}
        self.player_ids: list[str] = []
        self.queue_id = ""
        self.move_queues: list[deque[tuple[int, int, bool]]] = []

    def random_connected_map(
        self,
        seed: Any = None,
    ) -> None:
        g = self.graph
        es = g.es
        vs = g.vs
        rng = Random(seed)
        es["weight"] = [rng.random() for _ in range(g.ecount())]
        mst = g.spanning_tree("weight")
        leaves = mst.vs.select(_degree_le=1).indices
        g.delete_edges(_source=leaves)
        vs[leaves]["terrain"] = -2

    def set_generals(self, generals: list[int]) -> None:
        """Sets the generals on the map.

        Args:
            generals: The list of generals.

        Returns:
            None
        """
        graph = self.graph
        vs = graph.vs
        for i, v in enumerate(vs[generals]):
            v["terrain"] = i
        self.generals = generals
        self.move_queues = [deque() for _ in generals]

    def dispersion_generals(
        self,
        n: int,
        *,
        d_min: int = 9,
        seed: Any = None,
        verbose: bool = False,
    ) -> None:
        """Randomly selects generals on the map.

        Args:
            n: The number of generals to select.

        Keyword Args:
            d_min: The minimum distance between any two generals.
            seed: The seed for the random number generator.
            verbose: A flag indicating whether to print the solver status.

        Returns:
            None
        """
        g = self.graph
        vs = g.vs

        # sample vertices
        rng = Random(seed)
        indices = vs.select(_degree_gt=0).indices
        k = min(max(24, n * 12), len(indices))
        selects = rng.sample(indices, k)

        # update generals
        d_max = self.row + self.col - 2
        coords = vs[selects]["coord"]
        generals = p_dispersion(n, d_min, d_max, selects, coords, verbose)
        self.set_generals(generals)

    # ============================================================
    # send
    # ============================================================

    @to_task
    async def stars_and_rank(self, client: ClientProtocol, user_id: str) -> None:
        pass

    @to_task
    async def set_username(
        self, client: ClientProtocol, user_id: str, username: str
    ) -> None:
        self.usernames[user_id] = username

    @to_task
    async def join_private(
        self, client: ClientProtocol, queue_id: str, user_id: str
    ) -> None:
        self.queue_id = queue_id
        self.clients[client] = user_id
        self.player_ids.append(user_id)
        asyncio.create_task(self.run())

    @to_task
    async def set_force_start(
        self, client: ClientProtocol, queue_id: str, do_force: bool
    ) -> None:
        pass

    @to_task
    async def leave_game(self, client: ClientProtocol) -> None:
        user_id = self.clients[client]
        self.player_ids.remove(user_id)

    @to_task
    async def surrender(self, client: ClientProtocol) -> None:
        pass

    @to_task
    async def attack(
        self, client: ClientProtocol, start: int, end: int, is50: bool
    ) -> None:
        try:
            user_id = self.clients[client]
            i = self.player_ids.index(user_id)
        except (KeyError, ValueError):
            pass
        else:
            self.move_queues[i].append((start, end, is50))

    # ============================================================
    # recieve
    # ============================================================

    async def game_update(self) -> None:
        g = self.graph
        vs = g.vs

        turn = self.turn + 1

        # move armies
        for i, q in enumerate(self.move_queues):
            while q:
                start, end, is50 = q.popleft()
                if move_army(g, i, start, end, is50):
                    break

        # update generals and cities
        if turn % 2 == 1:
            for v in vs[self.generals]:
                v["army"] += 1
            for v in vs[self.cities]:
                v["army"] += 1

        # update lands
        if turn > 1 and turn % 50 == 1:
            for v in vs.select(terrain_ge=0):
                v["army"] += 1

        map_ = [self.col, self.row]
        map_ += vs["army"]
        map_ += vs["terrain"]

        map_diff = make_diff(map_, self.map_)
        self.map_ = map_
        self.turn = turn

        for c in self.clients.keys():
            c.game_update(
                {
                    "scores": [],
                    "turn": turn,
                    "stars": [],
                    "attackIndex": 0,
                    "generals": self.generals,
                    "map_diff": map_diff,
                    "cities_diff": [],
                }
            )

    # ============================================================
    # recieve
    # ============================================================

    # ============================================================
    # run
    # ============================================================

    async def connect(self, client: ClientProtocol) -> None:
        self.clients[client] = ""

    async def disconnect(self, client: ClientProtocol) -> None:
        del self.clients[client]

    async def run(self) -> None:
        self.random_connected_map(0)
        self.dispersion_generals(2)

        for c in self.clients.keys():
            c.game_start({})  # type: ignore

        start = time.monotonic()

        for _ in range(51):
            await self.game_update()

            end = time.monotonic()
            await asyncio.sleep(max(0, 0.5 - (end - start)))
            start = end

        for c in self.clients.keys():
            c.game_over()


class SocketioServer:
    def __init__(self) -> None:
        self.sockets: dict[ClientProtocol, AsyncClient] = {}

    # ============================================================
    # send
    # ============================================================

    def set_username(
        self, client: ClientProtocol, user_id: str, username: str
    ) -> asyncio.Task[None]:
        return asyncio.create_task(
            self.sockets[client].emit("set_username", (user_id, username, BOTKEY))
        )

    def stars_and_rank(
        self, client: ClientProtocol, user_id: str
    ) -> asyncio.Task[None]:
        return asyncio.create_task(self.sockets[client].emit("stars_and_rank", user_id))

    def join_private(
        self, client: ClientProtocol, queue_id: str, user_id: str
    ) -> asyncio.Task[None]:
        print(f"https://generals.io/games/{queue_id}")
        return asyncio.create_task(
            self.sockets[client].emit("join_private", (queue_id, user_id, BOTKEY))
        )

    def set_force_start(
        self, client: ClientProtocol, queue_id, do_force: bool
    ) -> asyncio.Task[None]:
        return asyncio.create_task(
            self.sockets[client].emit("set_force_start", (queue_id, do_force))
        )

    def leave_game(self, client: ClientProtocol) -> asyncio.Task[None]:
        return asyncio.create_task(self.sockets[client].emit("leave_game"))

    def surrender(self, client: ClientProtocol) -> asyncio.Task[None]:
        return asyncio.create_task(self.sockets[client].emit("surrender"))

    def attack(
        self, client: ClientProtocol, start: int, end: int, is50: bool
    ) -> asyncio.Task[None]:
        return asyncio.create_task(
            self.sockets[client].emit("attack", (start, end, is50))
        )

    # ============================================================
    # run
    # ============================================================

    async def connect(self, client: ClientProtocol) -> None:
        c = AsyncClient(ssl_verify=False)
        self.sockets[client] = c
        c.event(client.stars)
        c.event(client.rank)
        c.event(client.chat_message)
        c.event(client.notify)
        c.event(client.queue_update)
        c.event(client.pre_game_start)
        c.event(client.game_start)
        c.event(client.game_update)
        c.event(client.game_won)
        c.event(client.game_lost)
        c.event(client.game_over)
        await c.connect(WSURL, transports=["websocket"])

    async def disconnect(self, client: ClientProtocol) -> None:
        c = self.sockets[client]
        del self.sockets[client]
        await c.disconnect()
        await c.wait()

    async def run(self) -> None:
        pass

    # def draw_graph(self):
    #     """Draws the graph representation of the map.

    #     Returns:
    #         A tuple containing the figure and axis objects.
    #     """
    #     import matplotlib.pyplot as plt

    #     c, r = self.map_[:2]
    #     g = self.graph
    #     vs = g.vs
    #     fig, ax = plt.subplots(
    #         figsize=(c * 0.5, r * 0.5),
    #         layout="tight",
    #     )
    #     ax.invert_yaxis()
    #     ig.plot(
    #         g,
    #         ax,
    #         layout=vs["coord"],
    #         vertex_label=vs["name"],
    #         vertex_label_size=4,
    #     )
    #     return fig, ax
