from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from glhf.typing import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils.methods import asignalize, astreamify, methodlike

__all__ = [
    "ServerProtocol",
    "ClientProtocol",
    "HasGUI",
    "GUIProtocol",
    "BotProtocol",
    "Bot",
]


class ServerProtocol(Protocol):
    @abstractmethod
    async def connect(self, bot: BotProtocol) -> ClientProtocol: ...

    @abstractmethod
    async def disconnect(self, bot: BotProtocol) -> None: ...


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


class HasGUI(Protocol):
    gui: GUIProtocol | None


class GUIProtocol(Protocol):
    owner: HasGUI | None

    def is_registered(self) -> bool:
        return self.owner is not None

    def register(self, owner: HasGUI) -> None:
        if self.owner is not None:
            raise ValueError("GUI is already registered")
        if owner.gui is not None:
            raise ValueError("Owner already has a GUI")
        owner.gui = self
        self.owner = owner

    def deregister(self) -> None:
        if self.owner is None:
            raise ValueError("GUI is not registered")
        self.owner.gui = None
        self.owner = None

    @abstractmethod
    def game_start(self, data: GameStartDict) -> None: ...

    @abstractmethod
    def game_update(self, data: GameUpdateDict) -> None: ...

    @abstractmethod
    def game_over(self) -> None: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def __enter__(self) -> GUIProtocol:
        if not self.is_registered():
            raise ValueError("GUI is not registered")

        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()


@dataclass
class BotProtocol(Protocol):
    id: str
    name: str
    default_room: str = ""
    gui: GUIProtocol | None = None

    @methodlike
    def stars(self, data: dict[str, float]) -> Any:
        pass

    @methodlike
    def rank(self, data: dict[str, int]) -> Any:
        pass

    @methodlike
    def chat_message(self, chat_room: str, data: dict[str, Any]) -> Any:
        pass

    @methodlike
    def notify(self, data: Any) -> Any:
        pass

    @methodlike
    def queue_update(self, data: QueueUpdateDict) -> Any:
        pass

    @methodlike
    def pre_game_start(self) -> Any:
        pass

    @methodlike
    def game_start(self, data: GameStartDict) -> Any:
        pass

    @methodlike
    def game_update(self, data: GameUpdateDict) -> Any:
        pass

    @methodlike
    def game_won(self) -> Any:
        pass

    @methodlike
    def game_lost(self) -> Any:
        pass

    @methodlike
    def game_over(self) -> Any:
        pass

    async def start(self, server: ServerProtocol) -> None:
        try:
            client = await server.connect(self)
            await self.run(client)
        finally:
            await server.disconnect(self)

    @abstractmethod
    async def run(self, client: ClientProtocol) -> None: ...

    def __hash__(self) -> int: ...


@dataclass(unsafe_hash=True)
class Bot(BotProtocol):
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
    default_room: str = ""
    gui: GUIProtocol | None = field(repr=False, default=None)

    # ============================================================
    # recieve
    # ============================================================

    @astreamify
    def queue_update(self, data: QueueUpdateDict) -> QueueUpdateDict:
        return data

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
