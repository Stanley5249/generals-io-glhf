from typing import Any

from socketio import AsyncClient

from glhf.base import Agent, ClientProtocol
from glhf.typing import GameStartDict, GameUpdateDict, QueueUpdateDict
from glhf.utils.method import to_task

WSURL = "wss://ws.generals.io"
BOTKEY = "sd09fjd203i0ejwi_changeme"


class SocketioClient(ClientProtocol):
    def __init__(self, agent: Agent, socket: AsyncClient) -> None:
        self._agent = agent
        self._socket = socket
        self._queue_id = ""

    # ============================================================
    # recieve
    # ============================================================

    def stars(self, data: dict[str, float]) -> None:
        self._agent.stars(data)

    def rank(self, data: dict[str, int]) -> None:
        self._agent.rank(data)

    def chat_message(self, chat_room: str, data: dict[str, Any]) -> None:
        self._agent.chat_message(chat_room, data)

    def notify(self, data: Any, _: Any = None) -> None:
        self._agent.notify(data)

    def queue_update(self, data: dict) -> None:
        self._agent.queue_update(QueueUpdateDict(**data))

    def pre_game_start(self) -> None:
        self._agent.pre_game_start()

    def game_start(self, data: dict, _: Any = None) -> None:
        self._agent.game_start(GameStartDict(**data))

    def game_update(self, data: dict, _: Any = None) -> None:
        self._agent.game_update(GameUpdateDict(**data))

    def game_won(self, _1: Any = None, _2: Any = None) -> None:
        self._agent.game_won()

    def game_lost(self, _1: Any = None, _2: Any = None) -> None:
        self._agent.game_lost()

    def game_over(self, _1: Any = None, _2: Any = None) -> None:
        self._agent.game_over()

    # ============================================================
    # send
    # ============================================================

    @to_task
    async def set_username(self) -> None:
        await self._socket.emit(
            "set_username", (self._agent.id, self._agent.name, BOTKEY)
        )

    @to_task
    async def stars_and_rank(self) -> None:
        await self._socket.emit("stars_and_rank", self._agent.id)

    @to_task
    async def join_private(self, queue_id: str) -> None:
        self._queue_id = queue_id
        await self._socket.emit("join_private", (queue_id, self._agent.id, BOTKEY))
        print(f"https://generals.io/games/{queue_id}")

    @to_task
    async def set_force_start(self, do_force: bool) -> None:
        if not self._queue_id:
            raise RuntimeError("queue_id not set")
        await self._socket.emit("set_force_start", (self._queue_id, do_force))

    @to_task
    async def leave_game(self) -> None:
        await self._socket.emit("leave_game")

    @to_task
    async def surrender(self) -> None:
        await self._socket.emit("surrender")

    @to_task
    async def attack(self, start: int, end: int, is50: bool) -> None:
        await self._socket.emit("attack", (start, end, is50))


class SocketioServer:
    def __init__(self) -> None:
        self._sockets: dict[Agent, AsyncClient] = {}

    # ============================================================
    # run
    # ============================================================

    async def connect(self, agent: Agent) -> SocketioClient:
        socket = AsyncClient()
        client = SocketioClient(agent, socket)
        socket.event(client.stars)
        socket.event(client.rank)
        socket.event(client.chat_message)
        socket.event(client.notify)
        socket.event(client.queue_update)
        socket.event(client.pre_game_start)
        socket.event(client.game_start)
        socket.event(client.game_update)
        socket.event(client.game_won)
        socket.event(client.game_lost)
        socket.event(client.game_over)
        await socket.connect(WSURL, transports=["websocket"])
        self._sockets[agent] = socket
        return client

    async def disconnect(self, agent: Agent) -> None:
        socket = self._sockets[agent]
        await socket.disconnect()
        await socket.wait()
        del self._sockets[agent]
