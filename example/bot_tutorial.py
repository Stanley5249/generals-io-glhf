from glhf.base import ClientProtocol
from glhf.bot import Bot
from glhf.cli import main


class MyBot(Bot):
    async def run(self, client: ClientProtocol) -> None:
        client.join_private("example")

        async for data in self.queue_update:
            if not data["isForcing"]:
                client.set_force_start(True)

        async for data in self.game_update:
            if data["turn"] == 10:
                client.surrender()

        if self.game_won.get():
            print("I won!")
        elif self.game_lost.get():
            print("I lost!")
        else:
            print("Never!")


if __name__ == "__main__":
    main()
