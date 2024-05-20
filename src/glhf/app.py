import asyncio
import importlib.util
import pathlib
import sys
from contextlib import nullcontext
from inspect import isabstract
from typing import Any, Literal, Sequence

import fire
from fire.core import FireExit
from rich.console import Console
from rich.style import Style
from rich.text import Text

from glhf.base import BotProtocol, GUIProtocol, ServerProtocol
from glhf.gui import PygameGUI
from glhf.server import LocalServer, SocketioServer


def all_subclasses(cls: type[Any]) -> list[type[Any]]:
    ls = []
    for subclass in cls.__subclasses__():
        ls.append(subclass)
        ls.extend(all_subclasses(subclass))
    return ls


class APP:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

        self._bot_factory_list: list[type[BotProtocol]] = [
            b for b in all_subclasses(BotProtocol) if not isabstract(b)
        ]

        if not self._bot_factory_list:
            self._console.print("No bots found in module!", style="bold red")
            raise EOFError()  # exit temporarily

        self._bot_factory_dict = {b.__name__: b for b in self._bot_factory_list}

        self._bot_list: list[BotProtocol] = []

        self._gui_target: int | None = None
        self._gui_type: str = "pygame"

        self._server: ServerProtocol | None = None

        self._server_options: dict[str, type[ServerProtocol]] = {
            "local": LocalServer,
            "socketio": SocketioServer,
        }
        self._gui_options: dict[str, type[GUIProtocol]] = {
            "pygame": PygameGUI,
        }

    def show(self) -> None:
        self._console.print(self._gui_type, self._server, self._bot_list)

    def start(self, debug: bool = False) -> None:
        if self._server is None:
            self._console.print(
                f"No server set. Use the '{self.server.__name__}' command to set the server type."
            )
            return
        if not self._bot_list:
            self._console.print(
                f"No bots added. Use the '{self.bot_add.__name__}' command to add a bot."
            )
            return

        if self._gui_target is not None:
            gui = self._gui_options[self._gui_type]()
            gui.register(self._bot_list[self._gui_target])
        else:
            gui = None

        try:
            coro = start(self._server, self._bot_list, gui)

            loop = asyncio.get_event_loop()

            if loop.is_running():
                loop.set_debug(debug)
                task = loop.create_task(coro)
                task.add_done_callback(lambda t: t.result())

            else:
                asyncio.run(coro, debug=debug)

        finally:
            self._gui_target = None
            self._server = None
            self._bot_list = []

    def gui(self, index: int | None = None, name: str = "pygame") -> None:
        if index is None:
            self._gui_target = index
            self._console.print("gui cleared")
            return
        if name not in self._gui_options:
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
            server_type = self._server_options[name]
        except KeyError:
            self._console.print(f"unknown server: {name}")
            return
        try:
            self._server = server_type(*args, **kwargs)
        except TypeError:
            self._console.print_exception()

    def bot_help(self) -> None:
        self._console.print("Available bots:")
        for i, key in enumerate(self._bot_factory_dict):
            self._console.print(f"{i} {key!r}")

    def bot_show(self) -> None:
        for i, bot in enumerate(self._bot_list):
            self._console.print(f"{i}: {bot}")

    def bot_add(
        self,
        key: str | int,
        userid: str,
        username: str,
        default_room: str = "",
    ) -> None:
        if isinstance(key, int):
            if 0 <= key < len(self._bot_factory_list):
                bot_factory = self._bot_factory_list[key]
            else:
                self._console.print(f"unknown bot i-key: {key!r}", style="bold red")
                return
        else:
            if key in self._bot_factory_dict:
                bot_factory = self._bot_factory_dict[key]
            else:
                self._console.print(f"unknown bot key: {key!r}", style="bold red")
                return
        try:
            bot = bot_factory(userid, username, default_room)
        except Exception:
            self._console.print_exception()
            return

        self._bot_list.append(bot)
        self._console.print(f"+ {bot}", style="bold green")

    def bot_remove(self, index: int) -> None:
        if 0 <= index < len(self._bot_list):
            bot = self._bot_list.pop(index)
            self._console.print(f"- {bot}", style="bold green")
        else:
            self._console.print("index out of range", style="bold red")
            return

    def bot_clear(self) -> None:
        for bot in self._bot_list:
            self._console.print(f"- {bot}", style="bold green")
        self._bot_list.clear()


def set_eager_task_factory(is_eager: bool) -> None:
    loop = asyncio.get_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory if is_eager else None)  # type: ignore


async def start(
    server: ServerProtocol, bots: Sequence[BotProtocol], gui: GUIProtocol | None
) -> None:
    set_eager_task_factory(True)

    with gui or nullcontext():
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
