import asyncio
from typing import Any
from uuid import uuid4

from socketio import AsyncClient

from glhf.base import BotProtocol, ClientProtocol, ServerProtocol
from glhf.gui import PygameGUI
from glhf.typing_ import GameStartDict, GameUpdateDict, QueueUpdateDict

WSURL = "wss://ws.generals.io"
BOTKEY = "sd09fjd203i0ejwi_changeme"


class BasicClient(ClientProtocol):
    def __init__(
        self,
        userid: str,
        username: str,
        bot: BotProtocol,
        gui: PygameGUI,
        server: ServerProtocol,
    ) -> None:
        self.userid = userid
        self.username = username
        self.queue_id = ""
        self.server = server
        self.bot = bot
        self.gui = gui
        self.uuid = uuid4()

    # ============================================================
    # recieve
    # ============================================================

    def stars(self, data: dict[str, float]) -> None:
        self.bot.stars(data)

    def rank(self, data: dict[str, int]) -> None:
        self.bot.rank(data)

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> None:
        self.bot.chat_message(chat_room, data)

    def notify(self, data: Any, _: Any = None) -> None:
        self.bot.notify(data)

    def queue_update(self, data: QueueUpdateDict) -> None:
        self.bot.queue_update(data)

    def pre_game_start(self) -> None:
        self.bot.pre_game_start()

    def game_start(self, data: GameStartDict, _: Any = None) -> None:
        self.bot.game_start(data)
        self.gui.game_start(data)

    def game_update(self, data: GameUpdateDict, _: Any = None) -> None:
        self.bot.game_update(data)
        self.gui.game_update(data)

    def game_won(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_won()

    def game_lost(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_lost()

    def game_over(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_over()

    # ============================================================
    # send
    # ============================================================

    def set_username(self) -> asyncio.Task[None]:
        return self.server.set_username(self, self.userid, self.username)

    def stars_and_rank(self) -> asyncio.Task[None]:
        return self.server.stars_and_rank(self, self.userid)

    def join_private(self, queue_id: str) -> asyncio.Task[None]:
        self.queue_id = queue_id
        return self.server.join_private(self, queue_id, self.userid)

    def set_force_start(self, do_force: bool) -> asyncio.Task[None]:
        assert self.queue_id
        return self.server.set_force_start(self, self.queue_id, do_force)

    def leave_game(self) -> asyncio.Task[None]:
        return self.server.leave_game(self)

    def surrender(self) -> asyncio.Task[None]:
        return self.server.surrender(self)

    def attack(self, start: int, end: int, is50: bool) -> asyncio.Task[None]:
        return self.server.attack(self, start, end, is50)

    # ============================================================
    # run
    # ============================================================

    def __hash__(self) -> int:
        return hash(self.uuid)

    async def __aenter__(self) -> None:
        self.gui.__enter__()
        await self.server.connect(self)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.gui.__exit__(exc_type, exc_val, exc_tb)
        await self.server.disconnect(self)

    async def run(self) -> None:
        async with self:
            await self.bot.run(self)


class SocketioClient(AsyncClient, ClientProtocol):
    def __init__(
        self,
        userid: str,
        username: str,
        bot: BotProtocol,
        gui: PygameGUI,
        /,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs, ssl_verify=False)

        self.userid = userid
        self.username = username
        self.queue_id = ""
        self.bot = bot
        self.gui = gui

        self.event(self.stars)
        self.event(self.rank)
        self.event(self.chat_message)
        self.event(self.notify)
        self.event(self.queue_update)
        self.event(self.pre_game_start)
        self.event(self.game_start)
        self.event(self.game_update)
        self.event(self.game_won)
        self.event(self.game_lost)
        self.event(self.game_over)

    # ============================================================
    # recieve
    # ============================================================

    def stars(self, data: dict[str, float]) -> None:
        self.bot.stars(data)

    def rank(self, data: dict[str, int]) -> None:
        self.bot.rank(data)

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> None:
        self.bot.chat_message(chat_room, data)

    def notify(self, data: Any, _: Any = None) -> None:
        self.bot.notify(data)

    def queue_update(self, data: QueueUpdateDict) -> None:
        self.bot.queue_update(data)

    def pre_game_start(self) -> None:
        self.bot.pre_game_start()

    def game_start(self, data: GameStartDict, _: Any = None) -> None:
        self.bot.game_start(data)
        self.gui.game_start(data)

    def game_update(self, data: GameUpdateDict, _: Any = None) -> None:
        self.bot.game_update(data)
        self.gui.game_update(data)

    def game_won(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_won()

    def game_lost(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_lost()

    def game_over(self, _1: Any = None, _2: Any = None) -> None:
        self.bot.game_over()

    # ============================================================
    # emit
    # ============================================================

    def set_username(self) -> asyncio.Task[None]:
        return asyncio.create_task(
            self.emit("set_username", (self.userid, self.username, BOTKEY))
        )

    def stars_and_rank(self) -> asyncio.Task[None]:
        return asyncio.create_task(self.emit("stars_and_rank", self.userid))

    def join_private(self, queue_id: str) -> asyncio.Task[None]:
        self.queue_id = queue_id
        print(f"https://generals.io/games/{queue_id}")
        return asyncio.create_task(
            self.emit("join_private", (queue_id, self.userid, BOTKEY))
        )

    def set_force_start(self, do_force: bool) -> asyncio.Task[None]:
        assert self.queue_id
        return asyncio.create_task(
            self.emit("set_force_start", (self.queue_id, do_force))
        )

    def leave_game(self) -> asyncio.Task[None]:
        return asyncio.create_task(self.emit("leave_game"))

    def surrender(self) -> asyncio.Task[None]:
        return asyncio.create_task(self.emit("surrender"))

    def attack(self, start: int, end: int, is50: bool) -> asyncio.Task[None]:
        return asyncio.create_task(self.emit("attack", (start, end, is50)))

    # ============================================================
    # run
    # ============================================================

    async def __aenter__(self) -> None:
        self.gui.__enter__()
        await self.connect(WSURL, transports=["websocket"])

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.gui.__exit__(exc_type, exc_val, exc_tb)
        await self.disconnect()
        await self.wait()

    async def run(self) -> None:
        async with self:
            await self.bot.run(self)
