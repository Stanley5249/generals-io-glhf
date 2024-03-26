import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations as C
from itertools import product as P
from random import Random
from typing import Any

import igraph as ig
import ortools.sat.python.cp_model as cp
from bidict import KeyDuplicationError, ValueDuplicationError, bidict
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
    # vs["is_city"] = False
    # vs["is_general"] = False
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


@dataclass(slots=True)
class Queue:
    id: str = ""
    force_start: bool = False

    def clear(self) -> None:
        self.id = ""
        self.force_start = False


@dataclass(slots=True)
class Game:
    id: str = ""
    left: bool = False
    surrendered: bool = False
    attacks: deque[tuple[int, int, bool]] = field(default_factory=deque)


@dataclass(slots=True)
class User:
    id: str
    name: str = "Anonymous"


@dataclass(slots=True)
class Player:
    client: ClientProtocol
    user: User
    queue: Queue = field(default_factory=Queue)
    game: Game = field(default_factory=Game)


class LocalServer(ServerProtocol):
    def __init__(
        self,
        row: int,
        col: int,
        moves_per_turn: int = 2,
        turns_per_round: int = 25,
        turns_per_sec: float = 1.0,
    ) -> None:
        self.row = row
        self.col = col

        self.user_ids: bidict[ClientProtocol, str] = bidict()
        self.clients: bidict[str, ClientProtocol] = self.user_ids.inverse

        self.users: dict[str, User] = {}
        self.players: dict[str, Player] = {}
        self.game_queues: dict[str, list[Player]] = {}

        # settings
        self.moves_per_turn = moves_per_turn
        self.turns_per_round = turns_per_round
        self.turns_per_sec = turns_per_sec

    def random_connected_map(
        self,
        graph: ig.Graph,
        seed: Any = None,
    ) -> None:
        es = graph.es
        vs = graph.vs
        rng = Random(seed)
        es["weight"] = [rng.random() for _ in range(graph.ecount())]
        mst = graph.spanning_tree("weight")
        leaves = mst.vs.select(_degree_le=1).indices
        graph.delete_edges(_source=leaves)
        vs[leaves]["terrain"] = -2

    def set_generals(self, graph: ig.Graph, generals: list[int]) -> None:
        vs = graph.vs
        for i, v in enumerate(vs[generals]):
            v["terrain"] = i

    def dispersion_generals(
        self,
        n: int,
        row: int,
        col: int,
        graph: ig.Graph,
        *,
        d_min: int = 9,
        seed: Any = None,
        verbose: bool = False,
    ) -> list[int]:
        vs = graph.vs

        # sample vertices
        rng = Random(seed)
        indices = vs.select(_degree_gt=0).indices
        k = min(max(24, n * 12), len(indices))
        selects = rng.sample(indices, k)

        # update generals
        d_max = row + col - 2
        coords = vs[selects]["coord"]
        generals = p_dispersion(n, d_min, d_max, selects, coords, verbose)
        self.set_generals(graph, generals)
        return generals

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
        if self.user_ids[client] != user_id:
            raise
        self.users[user_id].name = username

    @to_task
    async def join_private(
        self, client: ClientProtocol, queue_id: str, user_id: str
    ) -> None:
        if self.user_ids[client] != user_id:
            raise
        game_queue = self.game_queues.setdefault(queue_id, [])
        player = self.players[user_id]
        player.queue.id = queue_id
        game_queue.append(player)
        self._queue_update(queue_id)

    @to_task
    async def set_force_start(
        self, client: ClientProtocol, queue_id: str, do_force: bool
    ) -> None:
        try:
            user_id = self.user_ids[client]
        except KeyError:
            raise
        players = self.game_queues[queue_id]
        for player in players:
            if player.user.id == user_id:
                player.queue.force_start = do_force
                break
        else:
            raise
        self._queue_update(queue_id)

    @to_task
    async def leave_game(self, client: ClientProtocol) -> None:
        try:
            user_id = self.user_ids[client]
        except KeyError:
            raise
        player = self.players[user_id]
        if player.game.id:
            player.game.left = True
        else:
            raise

    @to_task
    async def surrender(self, client: ClientProtocol) -> None:
        try:
            user_id = self.user_ids[client]
        except KeyError:
            raise
        player = self.players[user_id]
        if player.game.id:
            player.game.surrendered = True
        else:
            raise

    @to_task
    async def attack(
        self, client: ClientProtocol, start: int, end: int, is50: bool
    ) -> None:
        try:
            user_id = self.user_ids[client]
        except KeyError:
            raise
        player = self.players[user_id]
        player.game.attacks.append((start, end, is50))

    # ============================================================
    # recieve
    # ============================================================

    def _queue_update(self, queue_id: str) -> None:
        try:
            players = self.game_queues[queue_id]
        except KeyError:
            raise

        for i, player in enumerate(players):
            data = {
                "playerIndices": i,
                "isForcing": player.queue.force_start,
                "usernames": [player.user.name for player in players],
            }
            player.client.queue_update(data)  # type: ignore

        if sum(user_data.queue.force_start for user_data in players) >= 1:
            del self.game_queues[queue_id]
            for player in players:
                player.queue.clear()
                player.game.id = queue_id
            asyncio.create_task(self.run(players))

    def _game_start(self, players: list[Player]) -> None:
        for i, player in enumerate(players):
            player.client.game_start({"playerIndex": i})  # type: ignore

    def _game_update(self, players: list[Player], turn, generals, map_diff) -> None:
        for player in players:
            player.client.game_update(
                {
                    "scores": [],
                    "turn": turn,
                    "stars": [],
                    "attackIndex": 0,
                    "generals": generals,
                    "map_diff": map_diff,
                    "cities_diff": [],
                }
            )

    def _game_over(self, players: list[Player]) -> None:
        for player in players:
            player.client.game_over()

    def _map_update(
        self,
        players: list[Player],
        row: int,
        col: int,
        turn: int,
        map_old: list[int],
        generals: list[int],
        cities: list[int],
        graph: ig.Graph,
    ) -> tuple[list[int], list[int]]:
        vs = graph.vs

        # move armies
        for i, player in enumerate(players):
            attacks = player.game.attacks
            while attacks:
                start, end, is50 = attacks.popleft()
                if move_army(graph, i, start, end, is50):
                    break

        # update generals and cities
        if turn % 2 == 1:
            for v in vs[generals]:
                v["army"] += 1
            for v in vs[cities]:
                v["army"] += 1

        # update lands
        if turn > 1 and turn % 50 == 1:
            for v in vs.select(terrain_ge=0):
                v["army"] += 1

        map_new = [col, row]
        map_new += vs["army"]
        map_new += vs["terrain"]

        map_diff = make_diff(map_new, map_old)

        return map_new, map_diff

    # ============================================================
    # run
    # ============================================================

    async def connect(self, client: ClientProtocol, user_id: str) -> None:
        try:
            self.user_ids.put(client, user_id)
        except ValueDuplicationError:
            raise
        except KeyDuplicationError:
            raise
        if user_id not in self.users:
            self.users[user_id] = User(user_id)
        user = self.users[user_id]
        self.players[user_id] = Player(client, user)

    async def disconnect(self, client: ClientProtocol, user_id: str) -> None:
        if self.user_ids[client] != user_id:
            raise
        del self.user_ids[client]

    async def run(self, players: list[Player]) -> None:
        row = self.row
        col = self.col

        graph = make_2d_grid(row, col)
        self.random_connected_map(graph, 0)

        map_: list[int] = []
        generals: list[int] = self.dispersion_generals(2, row, col, graph)
        cities: list[int] = []

        t_start = time.monotonic()
        secs_per_move = self.turns_per_sec / self.moves_per_turn

        self._game_start(players)

        for turn in range(1, 52):
            map_, map_diff = self._map_update(
                players, row, col, turn, map_, generals, cities, graph
            )
            t_start += secs_per_move
            t_end = time.monotonic()
            await asyncio.sleep(max(0, t_start - t_end))
            self._game_update(players, turn, generals, map_diff)

            if __debug__:
                pass

        self._game_over(players)


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

    async def connect(self, client: ClientProtocol, user_id: str) -> None:
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

    async def disconnect(self, client: ClientProtocol, user_id: str) -> None:
        c = self.sockets[client]
        del self.sockets[client]
        await c.disconnect()
        await c.wait()

    async def run(self) -> None:
        pass
