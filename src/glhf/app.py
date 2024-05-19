import asyncio
import importlib.util
import pathlib
import sys
from typing import Literal, Sequence

import fire
from fire.core import FireExit
from rich.console import Console
from rich.style import Style
from rich.text import Text

from glhf.base import Bot, GUIProtocol, ServerProtocol
from glhf.gui import PygameGUI
from glhf.server import LocalServer, SocketioServer

USERID = "h4K1gOyHNnkGngym8fUuYA"
USERNAME = "PsittaTestBot"


class APP:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

        self._gui_target: int | None = None
        self._gui_type: str = "pygame"

        self._server: ServerProtocol | None = None
        self._bots: list[Bot] = []

        self._server_types: dict[str, type[ServerProtocol]] = {
            "local": LocalServer,
            "socketio": SocketioServer,
        }
        self._gui_types: dict[str, type[GUIProtocol]] = {"pygame": PygameGUI}
        self._bot_types = {bot.__name__: bot for bot in Bot.__subclasses__()}

        if not self._bot_types:
            self._console.print("No bots found in module!")

        self._bot_indices = tuple(self._bot_types.keys())

    def list(self) -> None:
        for i, k in enumerate(self._bot_indices):
            self._console.print(f"{i}: {k}")

    def show(self) -> None:
        self._console.print(self._gui_type, self._server, self._bots)

    def start(self, debug: bool = False) -> None:
        if self._server is None:
            self._console.print(
                f"No server set. Use the '{self.server.__name__}' command to set the server type."
            )
            return
        if not self._bots:
            self._console.print(
                f"No bots added. Use the '{self.add.__name__}' command to add a bot."
            )
            return
        try:
            if self._gui_target is not None:
                gui = self._gui_types[self._gui_type]()
                gui.connect()
                gui.register(self._bots[self._gui_target])
                asyncio.run(start(self._server, self._bots), debug=debug)
                gui.disconnect()
            else:
                asyncio.run(start(self._server, self._bots), debug=debug)
        finally:
            self._gui_target = None
            self._server = None
            self._bots = []

    def gui(self, index: int | None = None, name: str = "pygame") -> None:
        if index is None:
            self._gui_target = index
            self._console.print("gui cleared")
            return
        if name not in self._gui_types:
            self._console.print(f"unknown gui: {name}")
            return
        self._gui_target = index
        self._gui_type = name

    def server(
        self,
        name: Literal["local", "socketio"] = "local",
        *args,
        **kwargs,
    ) -> None:
        try:
            server_type = self._server_types[name]
        except KeyError:
            self._console.print(f"unknown server: {name}")
            return
        try:
            self._server = server_type(*args, **kwargs)
        except TypeError:
            self._console.print_exception()

    def add(
        self,
        key: str = "",
        index: int = 0,
        default_room: str = "",
        userid: str = USERID,
        username: str = USERNAME,
    ) -> None:
        if not key:
            try:
                key = self._bot_indices[index]
            except IndexError:
                self._console.print(f"unknown bot index: {index}")
                return
        try:
            bot_type = self._bot_types[key]
        except KeyError:
            self._console.print(f"unknown bot: {key}")
            return

        try:
            self._bots.append(bot_type(userid, username, default_room))
        except TypeError:
            self._console.print_exception()


def set_eager_task_factory(is_eager: bool) -> None:
    loop = asyncio.get_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory if is_eager else None)  # type: ignore


async def start(server: ServerProtocol, bots: Sequence[Bot]) -> None:
    set_eager_task_factory(True)

    async with asyncio.TaskGroup() as g:
        tasks = {g.create_task(bot.start(server)) for bot in bots}

    for task in tasks:
        task.result()


def command(file: str = "") -> None:
    if file:
        location = pathlib.Path(file)
        sys.path.append(str(location.parent.resolve()))
        name = location.stem
        spec = importlib.util.spec_from_file_location(name, location)
        if spec is None:
            raise IOError(f"could not load module from {location}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None, "spec.loader is None"
        spec.loader.exec_module(module)
    else:
        import __main__ as module

    console = Console()
    app = APP(console)

    name = "GLHF"
    prompt = Text.assemble(
        (name, Style(color="#008080", bold=True)),
        ("> ", Style(color="white", bold=True)),
    )

    while True:
        try:
            fire.Fire(app, console.input(prompt), name)

        except FireExit:
            pass

        except EOFError:
            console.print("GG!", style=Style(color="red", bold=True))
            break

        except Exception:
            console.print_exception()
            raise

        finally:
            console.print()
