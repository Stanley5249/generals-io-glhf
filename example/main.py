import asyncio

from glhf.bot import Bot
from glhf.client import BasicClient, SocketioClient
from glhf.gui import PygameGUI
from glhf.server import LocalServer, SocketioServer


async def main_basic() -> None:
    bot = Bot()
    gui = PygameGUI()
    server = SocketioServer()

    client = BasicClient(
        USERID,
        USERNAME,
        bot,
        gui,
        server,
    )
    await client.run()


async def main_local() -> None:
    bot = Bot()
    gui = PygameGUI()

    server = LocalServer(15, 15)

    client = BasicClient(
        USERID,
        USERNAME,
        bot,
        gui,
        server,
    )
    await client.run()


async def main_socketio() -> None:
    bot = Bot()
    gui = PygameGUI()
    client = SocketioClient(
        USERID,
        USERNAME,
        bot,
        gui,
    )
    await client.run()


if __name__ == "__main__":
    USERID = "123"
    USERNAME = "[BOT] 123"
    asyncio.run(main_socketio(), debug=False)
