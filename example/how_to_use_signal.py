import asyncio

from glhf.base import ClientProtocol
from glhf.bot import Bot
from glhf.client import SocketioClient
from glhf.gui import PygameGUI


class MyBot(Bot):
    async def run(self, client: ClientProtocol) -> None:
        client.join_private("signal-example")

        async for data in self.queue_update:
            if not data["isForcing"]:
                client.set_force_start(True)

        await self.game_over.wait()

        if self.game_won.get():
            print("I won!")
        elif self.game_lost.get():
            print("I lost!")
        else:
            print("Never!")


if __name__ == "__main__":
    USERID = "123"
    USERNAME = "[BOT] 123"

    bot = MyBot()
    gui = PygameGUI()

    client = SocketioClient(
        USERID,
        USERNAME,
        bot,
        gui,
    )

    asyncio.run(client.run(), debug=False)
