from __future__ import annotations

import time
from multiprocessing import Process, Queue
from typing import Any

from glhf.typing import GameStartDict, GameUpdateDict

__all__ = ["PygameGUI"]


class PygameGUI:
    def __init__(self) -> None:
        self.queue: Queue[Any] = Queue()
        self.process = Process(target=lazy_mainloop, args=(self.queue,))

    def game_start(self, data: GameStartDict) -> None:
        self.queue.put(("game_start", data))

    def game_update(self, data: GameUpdateDict) -> None:
        self.queue.put(("game_update", data))

    def game_over(self) -> None:
        self.queue.put(("game_over", None))

    def connect(self) -> None:
        self.process.start()
        time.sleep(2)

    def disconnect(self) -> None:
        self.queue.put(None)
        if self.process.is_alive():
            self.process.join()
        self.process.close()


def lazy_mainloop(queue: Queue[Any]) -> None:
    from glhf.gui._pygame.states import mainloop

    mainloop(queue)
