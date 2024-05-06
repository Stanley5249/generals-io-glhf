from __future__ import annotations

from functools import partial
from multiprocessing import Queue
from queue import Empty
from typing import Any, Generator, Protocol, Self

import pygame
import pygame.font

from glhf.gui._pygame.render import BackgroundSprite, MapSprite, Renderer
from glhf.gui._resource import FONT, IMG_CITY, IMG_CROWN, IMG_MOUNTAIN, IMG_OBSTACLE


class PygameState(Protocol):
    def __call__(self, window: pygame.Surface, queue: Queue[Any]) -> Self | None: ...


def get_stream[T](queue: Queue[T]) -> Generator[Any, Any, T | None]:
    while True:
        try:
            yield queue.get_nowait()
        except Empty:
            return


def get_data[T](queue: Queue[T]) -> T | None:
    try:
        return queue.get_nowait()
    except Empty:
        return


def game(window: pygame.Surface, queue: Queue[Any]) -> PygameState | None:
    renderer = Renderer(
        pygame.image.load(IMG_CITY).convert_alpha(),
        pygame.image.load(IMG_CROWN).convert_alpha(),
        pygame.image.load(IMG_MOUNTAIN).convert_alpha(),
        pygame.image.load(IMG_OBSTACLE).convert_alpha(),
        partial(pygame.font.Font, FONT),
    )

    clock = pygame.time.Clock()

    bg = BackgroundSprite()
    bg.fast_update(True)
    map_ = MapSprite(renderer=renderer)

    group = pygame.sprite.LayeredDirty(bg, map_)  # type: ignore
    rects = group.draw(window)
    pygame.display.update(rects)

    move = pygame.Vector2()

    while True:
        window_resize = False
        zoom = 0
        move.update()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
            elif e.type == pygame.WINDOWSIZECHANGED:
                window_resize = True
            elif e.type == pygame.MOUSEWHEEL:
                zoom += e.x + e.y
            elif e.type == pygame.MOUSEMOTION:
                if e.buttons[0] == 1:
                    move += e.rel

        data = get_data(queue)
        if data is not None:
            if data[0] == "game_update":
                data = data[1]
            else:
                data = None

        bg.fast_update(window_resize)
        map_.fast_update(zoom, move, data)

        rects = group.draw(window)
        if rects:
            pygame.display.update(rects)

        clock.tick(60)


def lobby(window: pygame.Surface, queue: Queue[Any]) -> PygameState | None:
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return

            match get_data(queue):
                case None:
                    continue
                case ("game_start", _data):
                    return game


def mainloop(queue: Queue[Any]) -> None:
    pygame.init()

    try:
        window = pygame.display.set_mode((800, 600))
        state: PygameState | None = lobby
        while state is not None:
            state = state(window, queue)

    finally:
        pygame.quit()
