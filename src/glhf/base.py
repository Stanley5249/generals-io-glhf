from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from glhf.gui import PygameGUI
from glhf.typing import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils.method import methodlike

__all__ = "ServerProtocol", "ClientProtocol", "BotProtocol", "Agent"


class ServerProtocol(Protocol):
    @abstractmethod
    async def connect(self, agent: Agent) -> ClientProtocol: ...

    @abstractmethod
    async def disconnect(self, agent: Agent) -> None: ...


class ClientProtocol(Protocol):
    # ============================================================
    # send
    # ============================================================

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


@dataclass(frozen=True, slots=True)
class Agent:
    id: str = field(compare=True)
    name: str = field(compare=True)
    bot: BotProtocol = field(compare=False)
    gui: PygameGUI | None = field(compare=False)

    # ============================================================
    # receive
    # ============================================================

    def stars(self, data: dict[str, float]) -> Any:
        self.bot.stars(data)
        # if self.gui:
        #     self.gui.stars(data)

    def rank(self, data: dict[str, int]) -> Any:
        self.bot.rank(data)
        # if self.gui:
        #     self.gui.rank(data)

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> Any:
        self.bot.chat_message(chat_room, data)
        # if self.gui:
        #     self.gui.chat_message(chat_room, data)

    def notify(self, data: Any) -> Any:
        self.bot.notify(data)
        # if self.gui:
        #     self.gui.notify(data)

    def queue_update(self, data: QueueUpdateDict) -> Any:
        self.bot.queue_update(data)
        # if self.gui:
        #     self.gui.queue_update(data)

    def pre_game_start(self) -> Any:
        self.bot.pre_game_start()
        # if self.gui:
        #     self.gui.pre_game_start()

    def game_start(self, data: GameStartDict) -> Any:
        self.bot.game_start(data)
        if self.gui:
            self.gui.game_start(data)

    def game_update(self, data: GameUpdateDict) -> Any:
        self.bot.game_update(data)
        if self.gui:
            self.gui.game_update(data)

    def game_won(self) -> Any:
        self.bot.game_won()
        # if self.gui:
        #     self.gui.game_won()

    def game_lost(self) -> Any:
        self.bot.game_lost()
        # if self.gui:
        #     self.gui.game_lost()

    def game_over(self) -> Any:
        self.bot.game_over()
        if self.gui:
            self.gui.game_over()

    async def run(self, server: ServerProtocol) -> None:
        client = await server.connect(self)
        try:
            if self.gui:
                with self.gui:
                    await self.bot.run(client)
            else:
                await self.bot.run(client)
        finally:
            await server.disconnect(self)


@runtime_checkable
class BotProtocol(Protocol):
    # ============================================================
    # recieve
    # ============================================================

    @methodlike
    @abstractmethod
    def stars(self, data: dict[str, float]) -> Any: ...

    @methodlike
    @abstractmethod
    def rank(self, data: dict[str, int]) -> Any: ...

    @methodlike
    @abstractmethod
    def chat_message(self, chat_room: str, data: dict[str, Any]) -> Any: ...

    @methodlike
    @abstractmethod
    def notify(self, data: Any) -> Any: ...

    @methodlike
    @abstractmethod
    def queue_update(self, data: QueueUpdateDict) -> Any: ...

    @methodlike
    @abstractmethod
    def pre_game_start(self) -> Any: ...

    @methodlike
    @abstractmethod
    def game_start(self, data: GameStartDict) -> Any: ...

    @methodlike
    @abstractmethod
    def game_update(self, data: GameUpdateDict) -> Any: ...

    @methodlike
    @abstractmethod
    def game_won(self) -> Any: ...

    @methodlike
    @abstractmethod
    def game_lost(self) -> Any: ...

    @methodlike
    @abstractmethod
    def game_over(self) -> Any: ...

    # ============================================================
    # run
    # ============================================================

    @abstractmethod
    async def run(self, client: ClientProtocol) -> None: ...


class GUIProtocol(Protocol): ...
