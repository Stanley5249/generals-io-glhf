from __future__ import annotations

from asyncio import Task, create_task
from typing import Any

from glhf.base import BotProtocol
from glhf.helper import asyncio_queueify
from glhf.typing_ import GameStartDict, GameUpdateDict, QueueUpdateDict


class Bot(BotProtocol):
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
