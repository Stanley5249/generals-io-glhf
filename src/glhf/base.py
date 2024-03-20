from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, Protocol

from glhf.typing_ import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils import methodlike

__all__ = "ServerProtocol", "ClientProtocol", "BotProtocol"


class ServerProtocol(Protocol):
    # ============================================================
    # send
    # ============================================================

    @abstractmethod
    def set_username(
        self,
        client: ClientProtocol,
        user_id: str,
        username: str,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def stars_and_rank(
        self,
        client: ClientProtocol,
        user_id: str,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def join_private(
        self,
        client: ClientProtocol,
        queue_id: str,
        user_id: str,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def set_force_start(
        self,
        client: ClientProtocol,
        queue_id: str,
        do_force: bool,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def leave_game(
        self,
        client: ClientProtocol,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def surrender(
        self,
        client: ClientProtocol,
    ) -> asyncio.Task[None]: ...

    @abstractmethod
    def attack(
        self, client: ClientProtocol, start: int, end: int, is50: bool
    ) -> asyncio.Task[None]: ...

    # ============================================================
    # run
    # ============================================================

    @abstractmethod
    async def connect(
        self,
        client: ClientProtocol,
    ) -> None: ...

    @abstractmethod
    async def disconnect(
        self,
        client: ClientProtocol,
    ) -> None: ...

    @abstractmethod
    async def run(self) -> None: ...


class ClientProtocol(Protocol):
    # ============================================================
    # recieve
    # ============================================================

    @abstractmethod
    def stars(self, data: dict[str, float]) -> Any: ...

    @abstractmethod
    def rank(self, data: dict[str, int]) -> Any: ...

    @abstractmethod
    def chat_message(self, chat_room: str, data: dict[str, Any]) -> Any: ...

    @abstractmethod
    def notify(self, data: Any, _: Any = None) -> Any: ...

    @abstractmethod
    def queue_update(self, data: QueueUpdateDict) -> Any: ...

    @abstractmethod
    def pre_game_start(self) -> Any: ...

    @abstractmethod
    def game_start(self, data: GameStartDict, _: Any = None) -> Any: ...

    @abstractmethod
    def game_update(self, data: GameUpdateDict, _: Any = None) -> Any: ...

    @abstractmethod
    def game_won(self, _1: Any = None, _2: Any = None) -> Any: ...

    @abstractmethod
    def game_lost(self, _1: Any = None, _2: Any = None) -> Any: ...

    @abstractmethod
    def game_over(self, _1: Any = None, _2: Any = None) -> Any: ...

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

    # ============================================================
    # run
    # ============================================================

    @abstractmethod
    async def run(self) -> None: ...


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
