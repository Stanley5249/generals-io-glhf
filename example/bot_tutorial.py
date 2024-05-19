from glhf.app import command
from glhf.base import Bot, ClientProtocol


class MyBot(Bot):
    async def run(self, client: ClientProtocol) -> None:
        client.join_private(self.default_room or "example")

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
    command()
