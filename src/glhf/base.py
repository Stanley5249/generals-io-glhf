from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from glhf.gui import PygameGUI
from glhf.typing import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils.methods import asignalize, astreamify

__all__ = ["ServerProtocol", "ClientProtocol", "Bot"]


class ServerProtocol(Protocol):
    @abstractmethod
    async def connect(self, bot: Bot) -> ClientProtocol: ...

    @abstractmethod
    async def disconnect(self, bot: Bot) -> None: ...


class ClientProtocol(Protocol):
    @abstractmethod
    def set_username(self) -> asyncio.Task[None]: ...

    @abstractmethod
    def stars_and_rank(self) -> asyncio.Task[None]: ...

    @abstractmethod
    def join_private(self, queue_id: str) -> asyncio.Task[None]: ...

    @abstractmethod
    def set_force_start(self, do_force: bool) -> asyncio.Task[None]: ...

    @abstractmethod
    def leave_game(self) -> asyncio.Task[None]: ...

    @abstractmethod
    def surrender(self) -> asyncio.Task[None]: ...

    @abstractmethod
    def attack(self, start: int, end: int, is50: bool) -> asyncio.Task[None]: ...


class GUIProtocol(Protocol): ...


@dataclass(frozen=True)
class Bot(ABC):
    """The `Bot` class allows customization by overriding the `run` method for specific game interactions.

    Subclassing `Bot` provides additional functionality:

    The `queue_update`, `game_start`, and `game_update` methods are enhanced with the `astreamify` decorator. This decorator transforms these methods into an async generator, which can be used in an async for loop for server update processing.

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
        - `astreamify`
        - `asignalize`

    """

    id: str
    name: str
    gui: PygameGUI | None = None

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

    @astreamify
    def queue_update(self, data: QueueUpdateDict) -> QueueUpdateDict:
        return data

    def pre_game_start(self) -> None:
        pass

    @astreamify
    def game_start(self, data: GameStartDict) -> GameStartDict:
        self.queue_update.close()
        if self.gui:
            self.gui.game_start(data)
        return data

    @astreamify
    def game_update(self, data: GameUpdateDict) -> GameUpdateDict:
        if self.gui:
            self.gui.game_update(data)
        return data

    @asignalize
    def game_won(self) -> None:
        pass

    @asignalize
    def game_lost(self) -> None:
        pass

    @asignalize
    def game_over(self) -> None:
        self.game_update.close()
        if self.gui:
            self.gui.game_over()

    async def start(self, server: ServerProtocol) -> None:
        try:
            client = await server.connect(self)
            if self.gui:
                self.gui.connect()
            await self.run(client)
        finally:
            if self.gui:
                self.gui.disconnect()
            await server.disconnect(self)

    @abstractmethod
    async def run(self, client: ClientProtocol) -> None: ...
