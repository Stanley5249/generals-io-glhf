from __future__ import annotations

from asyncio import Task, create_task
from typing import Any

from glhf.base import BotProtocol
from glhf.typing_ import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils import asyncio_queueify


class Bot(BotProtocol):
    """The `Bot` class is a subclass of `BotProtocol`. It allows customization by overriding the `run` method for specific game interactions. Alternatively, you can create a class that adheres to the `BotProtocol` interface.

    Subclassing `Bot` provides additional functionality:

    The `queue_update`, `game_start`, and `game_update` methods are enhanced with the `asyncio_queueify` decorator. This decorator transforms these methods into an async generator, which can be used in an async for loop for server update processing.

    Example:
        ```python
        class MyBot(Bot):
            async def run(self, client: ClientProtocol) -> None:
                ...
                # process queue updates
                async for data in self.queue_update:
                    if not data["isForcing"]:
                        client.set_force_start(True)

                # wait for game start
                await self.game_start.wait()

                # process game updates
                map_ = []
                cities = []
                async for data in self.game_update:
                    map_ = patch(map_, data["map_diff"])
                    cities = patch(cities, data["cities_diff"])
                ...
        ```

    See Also:
        - `BotProtocol`
        - `asyncio_queueify`

    """

    def __init__(self) -> None:
        self._tasks: set[Task[Any]] = set()

    # ============================================================
    # recieve
    # ============================================================

    def stars(self, data: dict[str, float]) -> None:
        pass

    def rank(self, data: dict[str, int]) -> None:
        pass

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> None:
        pass

    def notify(self, data: Any) -> None:
        pass

    @asyncio_queueify
    def queue_update(self, data: QueueUpdateDict) -> QueueUpdateDict:
        return data

    def pre_game_start(self) -> None:
        pass

    @asyncio_queueify
    def game_start(self, data: GameStartDict) -> GameStartDict:
        task = create_task(self.queue_update.put(None))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return data

    @asyncio_queueify
    def game_update(self, data: GameUpdateDict) -> GameUpdateDict:
        return data

    def game_won(self) -> None:
        pass

    def game_lost(self) -> None:
        pass

    def game_over(self) -> None:
        task = create_task(self.game_update.put(None))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
