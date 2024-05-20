from secrets import token_urlsafe

from glhf.app import APP
from glhf.base import Bot, ClientProtocol


class SurrenderBot(Bot):
    async def run(self, client: ClientProtocol) -> None:
        client.join_private(self.default_room or token_urlsafe(3))

        async for data in self.queue_update:
            if not data["isForcing"]:
                client.set_force_start(True)

        async for data in self.game_update:
            if data["turn"] == 1:
                client.surrender()

        if self.game_won.get():
            print("win")
        elif self.game_lost.get():
            print("lose")
        else:
            print("never")


def main() -> None:
    USERID = "123"
    USERNAME = "[Bot]123"
    app = APP()
    app.server("socketio")
    app.bot_add("SurrenderBot", userid=USERID, username=USERNAME)
    app.gui(0)
    app.start()


if __name__ == "__main__":
    main()
