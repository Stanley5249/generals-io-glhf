from __future__ import annotations

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

from glhf.base import Bot, ClientProtocol, ServerProtocol
from glhf.typing import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils.maps import make_diff
from glhf.utils.methods import to_task


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
    bot: Bot
    user: User
    queue: Queue = field(default_factory=Queue)
    game: Game = field(default_factory=Game)


class LocalClient(ClientProtocol):
    def __init__(self, bot: Bot, server: LocalServer) -> None:
        self._bot = bot
        self._server = server
        self._queue_id = ""

    # ============================================================
    # recieve
    # ============================================================

    def stars(self, data: dict[str, float]) -> None:
        self._bot.stars(data)

    def rank(self, data: dict[str, int]) -> None:
        self._bot.rank(data)

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> None:
        self._bot.chat_message(chat_room, data)

    def notify(self, data: Any) -> None:
        self._bot.notify(data)

    def queue_update(self, data: QueueUpdateDict) -> None:
        self._bot.queue_update(data)

    def pre_game_start(self) -> None:
        self._bot.pre_game_start()

    def game_start(self, data: GameStartDict) -> None:
        self._bot.game_start(data)

    def game_update(self, data: GameUpdateDict) -> None:
        self._bot.game_update(data)

    def game_won(self) -> None:
        self._bot.game_won()

    def game_lost(self) -> None:
        self._bot.game_lost()

    def game_over(self) -> None:
        self._bot.game_over()

    # ============================================================
    # send
    # ============================================================

    @to_task
    async def set_username(self) -> None:
        await self._server.set_username(self._bot)

    @to_task
    async def stars_and_rank(self) -> None:
        await self._server.stars_and_rank(self._bot)

    @to_task
    async def join_private(self, queue_id: str) -> None:
        self._queue_id = queue_id
        await self._server.join_private(self._bot, self._queue_id)
        print(queue_id)

    @to_task
    async def set_force_start(self, do_force: bool) -> None:
        if not self._queue_id:
            raise RuntimeError("queue_id not set")
        await self._server.set_force_start(self._bot, self._queue_id, do_force)

    @to_task
    async def leave_game(self) -> None:
        await self._server.leave_game(self._bot)

    @to_task
    async def surrender(self) -> None:
        await self._server.surrender(self._bot)

    @to_task
    async def attack(self, start: int, end: int, is50: bool) -> None:
        await self._server.attack(self._bot, start, end, is50)


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

        self.userids: bidict[Bot, str] = bidict()
        self.bots: bidict[str, Bot] = self.userids.inverse

        self.users: dict[str, User] = {}
        self.players: dict[str, Player] = {}
        self.game_queues: dict[str, list[Player]] = {}

        # settings
        self.mpt = moves_per_turn
        self.tpr = turns_per_round
        self.tps = turns_per_sec

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
    async def stars_and_rank(self, bot: Bot) -> None:
        pass

    @to_task
    async def set_username(self, bot: Bot) -> None:
        if self.userids[bot] != bot.id:
            raise ValueError("userid mismatch")
        self.users[bot.id].name = bot.name

    @to_task
    async def join_private(self, bot: Bot, queue_id: str) -> None:
        if self.userids[bot] != bot.id:
            raise ValueError("userid mismatch")
        game_queue = self.game_queues.setdefault(queue_id, [])
        player = self.players[bot.id]
        player.queue.id = queue_id
        game_queue.append(player)
        self._queue_update(queue_id)

    @to_task
    async def set_force_start(
        self, bot: Bot, queue_id: str, do_force: bool
    ) -> None:
        try:
            userid = self.userids[bot]
        except KeyError:
            raise
        players = self.game_queues[queue_id]
        for player in players:
            if player.user.id == userid:
                player.queue.force_start = do_force
                break
        else:
            raise
        self._queue_update(queue_id)

    @to_task
    async def leave_game(self, bot: Bot) -> None:
        try:
            userid = self.userids[bot]
        except KeyError:
            raise
        player = self.players[userid]
        if player.game.id:
            player.game.left = True
        else:
            raise

    @to_task
    async def surrender(self, bot: Bot) -> None:
        try:
            userid = self.userids[bot]
        except KeyError:
            raise
        player = self.players[userid]
        if player.game.id:
            player.game.surrendered = True
        else:
            raise

    @to_task
    async def attack(self, bot: Bot, start: int, end: int, is50: bool) -> None:
        try:
            userid = self.userids[bot]
        except KeyError:
            raise
        player = self.players[userid]
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
            player.bot.queue_update(data)  # type: ignore

        if sum(user_data.queue.force_start for user_data in players) >= 1:
            del self.game_queues[queue_id]
            for player in players:
                player.queue.clear()
                player.game.id = queue_id
            asyncio.create_task(self.run(players))

    def _game_start(self, players: list[Player]) -> None:
        for i, player in enumerate(players):
            player.bot.game_start({"playerIndex": i})  # type: ignore

    def _game_update(self, players: list[Player], turn, generals, map_diff) -> None:
        for player in players:
            player.bot.game_update(
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
            player.bot.game_over()

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

    async def connect(self, bot: Bot) -> LocalClient:
        try:
            self.userids.put(bot, bot.id)
        except ValueDuplicationError:
            raise RuntimeError(f"{bot.id} is already connected")
        except KeyDuplicationError:
            raise RuntimeError("Bot cannot be connected twice")
        client = LocalClient(bot, self)
        if bot.id not in self.users:
            self.users[bot.id] = User(bot.id)
        user = self.users[bot.id]
        self.players[bot.id] = Player(bot, user)
        return client

    async def disconnect(self, bot: Bot) -> None:
        if self.userids[bot] != bot.id:
            raise ValueError("userid mismatch")
        del self.userids[bot]

    async def run(self, players: list[Player]) -> None:
        row = self.row
        col = self.col

        graph = make_2d_grid(row, col)
        self.random_connected_map(graph, 0)

        map_: list[int] = []
        generals: list[int] = self.dispersion_generals(2, row, col, graph)
        cities: list[int] = []

        t_start = time.monotonic()
        secs_per_move = self.tps / self.mpt

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
