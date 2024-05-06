import asyncio

from secrets import token_urlsafe
from typing import Iterable, Sequence

import igraph as ig
from collections import deque
from glhf.base import ClientProtocol
from glhf.bot import Bot
from glhf.cli import main
from glhf.utils.maps import patch

from algorithm import make_graph, opening_moves

type AttackT = tuple[int, int, bool]
type AttackPathT = tuple[int, Sequence[int]]


class AttackScheduler:
    def __init__(self, iterable: Iterable[AttackPathT] | None = None) -> None:
        self.data = deque() if iterable is None else deque(iterable)

    def add_path(self, x: AttackPathT) -> None:
        self.data.append(x)

    def add_paths(self, iterable: Iterable[AttackPathT]) -> None:
        self.data.extend(iterable)

    def get(self, turn: int) -> AttackT | None:
        while self.data:
            start, path = self.data[0]
            i = turn - start
            if i < 0:
                break
            if i < len(path) - 1:
                return path[i], path[i + 1], False
            self.data.popleft()


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
        atk_sched = AttackScheduler()
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
                        verbose=True,
                    )
                )
                task.add_done_callback(
                    lambda t: atk_sched.add_paths(t.result())
                    if t.exception() is None
                    else t.print_stack()
                )

            elif turn == 50:
                client.surrender()

            atk = atk_sched.get(turn)
            if atk is not None:
                client.attack(*atk)

        client.leave_game()


if __name__ == "__main__":
    main()
