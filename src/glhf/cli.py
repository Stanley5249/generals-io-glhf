import asyncio
import importlib.util
import pathlib
import sys
import traceback
from types import ModuleType
from typing import Literal, Sequence

import fire
from fire.core import FireExit
from rich.console import Console
from rich.style import Style
from rich.text import Text

from glhf.client import BasicClient
from glhf.base import ClientProtocol, ServerProtocol
from glhf.gui import PygameGUI
from glhf.server import LocalServer, SocketioServer


class CLI:
    """Command Line Interface for managing clients and starting the server"""

    def __init__(self, module: ModuleType, console: Console) -> None:
        self._module = module
        self._console = console
        self.__lazy_init__()

    def __lazy_init__(self) -> None:
        self._server: ServerProtocol | None = None
        self._clients: list[ClientProtocol] = []

    def show(self) -> None:
        """Display the server and client information."""
        self._console.print(self._server, self._clients)

    def start(self, debug: bool = False) -> None:
        """Start the server.

        Args:
            debug (bool, optional): Whether to enable debug mode. Defaults to False.

        """
        if self._server is None:
            raise RuntimeError("server not set")
        asyncio.run(start(self._server, self._clients), debug=debug)
        self.__lazy_init__()

    def server(self, name: str | Literal["local", "socketio"] = "local") -> None:
        """Set the server type.

        Args:
            name (str | Literal["local", "socketio"], optional): The server type to set. Defaults to "socketio".

        Raises:
            ValueError: If the provided server name is unknown.

        """

        if name == "local":
            self._server = LocalServer(15, 15)
        elif name == "socketio":
            self._server = SocketioServer()
        else:
            try:
                cls = getattr(self._module, name)
            except AttributeError:
                raise ValueError(f"unknown server: {name}")
            else:
                self._server = cls()

    def client(self, bot: str) -> None:
        """Add a bot to the client list.

        Args:
            name (str): The name of the bot to add.

        Raises:
            ValueError: If the provided bot name is unknown.

        """

        if self._server is None:
            raise RuntimeError("server not set")

        try:
            bot_cls = getattr(self._module, bot)
        except AttributeError:
            raise ValueError(f"unknown bot: {bot}")
        else:
            bot_obj = bot_cls()
            USERID = "h4K1gOyHNnkGngym8fUuYA"
            USERNAME = "PsittaTestBot"
            gui = PygameGUI()
            client = BasicClient(USERID, USERNAME, bot_obj, gui, self._server)
            self._clients.append(client)


def set_eager_task_factory(is_eager: bool) -> None:
    loop = asyncio.get_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory if is_eager else None)  # type: ignore


async def start(server: ServerProtocol, clients: Sequence[ClientProtocol]) -> None:
    set_eager_task_factory(True)

    async with asyncio.TaskGroup() as g:
        for client in clients:
            g.create_task(client.run())


def main(file: str = "") -> None:

    if file:
        location = pathlib.Path(file)
        name = location.stem
        spec = importlib.util.spec_from_file_location(name, location)
        if spec is None:
            raise IOError(f"could not load module from {location}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None, "spec.loader is None"
        spec.loader.exec_module(module)
    else:
        # module = importlib.import_module('__main__')
        import __main__ as module

    console = Console()
    cli = CLI(module, console)

    name = "GLHF"
    prompt = Text.assemble(
        (name, Style(color="#008080", bold=True)),
        ("> ", Style(color="white", bold=True)),
    )

    while True:
        try:
            fire.Fire(cli, console.input(prompt), name)

        except FireExit:
            pass

        except EOFError:
            console.print("GG!", style="b red")
            break

        except Exception:
            traceback.print_exc()

        finally:
            console.print()


if __name__ == "__main__":
    main()
