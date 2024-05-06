from __future__ import annotations

import time
from functools import lru_cache
from itertools import product
from multiprocessing import Process, Queue, current_process
from os import PathLike
from pathlib import Path
from queue import Empty
from typing import IO, Any, Generator, Protocol, Self, Sequence

from glhf.gui._resource import FONT, IMG_CITY, IMG_CROWN, IMG_MOUNTAIN, IMG_OBSTACLE
from glhf.typing import GameStartDict, GameUpdateDict
from glhf.utils.maps import patch

type AnyPath = str | bytes | PathLike[str] | PathLike[bytes]
type FileArg = AnyPath | IO[bytes] | IO[str]
type Coordinate = Sequence[float]


RESDIR = Path(__file__).parent / "resource"
PALETTE_RAW = (
    0x222222FF,
    0x393939FF,
    0x808080FF,
    0xBBBBBBFF,
    0xDCDCDCFF,
    0xFF0000FF,
    0x4363D8FF,
    0x008000FF,
    0x008080FF,
    0xF58231FF,
    0xF032E6FF,
    0x800080FF,
    0x800000FF,
    0xB09F30FF,
    0x9A6324FF,
    0x0000FFFF,
    0x483D8BFF,
)

if current_process().name != "MainProcess":
    import pygame


class State(Protocol):
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


def game(window: pygame.Surface, queue: Queue[Any]) -> State | None:
    import pygame
    import pygame.font

    PALETTE = tuple(map(pygame.Color, PALETTE_RAW))
    WHITE = pygame.Color("white")
    BLACK = pygame.Color("black")
    # ARROWS = "↑↓←→"

    class CachedResource:
        __slots__ = "font_", "city", "crown", "mountain", "obstacle"

        def __init__(
            self,
            font_: FileArg = FONT,
            city: FileArg = IMG_CITY,
            crown: FileArg = IMG_CROWN,
            mountain: FileArg = IMG_MOUNTAIN,
            obstacle: FileArg = IMG_OBSTACLE,
        ) -> None:
            self.font_ = font_
            self.city = pygame.image.load(city).convert_alpha()
            self.crown = pygame.image.load(crown).convert_alpha()
            self.mountain = pygame.image.load(mountain).convert_alpha()
            self.obstacle = pygame.image.load(obstacle).convert_alpha()

        def render_cell(self, size: int, terrain: int, army: int) -> pygame.Surface:
            """
            Render the terrain without army.

            Args:
                size (int): The size of the cell.
                terrain (int): The terrain value.

            Returns:
                Surface: The rendered cell.

            Raises:
                ValueError: If the terrain value is not within the valid range.

            Notes:
                The meaning of different terrain values are as follows:
                -4: fog obstacle
                -3: fog
                -2: mountain
                -1: neutral without army
                0 ~ 11: lands
                12 ~ 23: cities
                24 ~ 35: crowns
                36: neutral with army

            """
            if army and terrain >= 0:
                return self.render_terrain_army(size, terrain, army)
            return self.render_terrain(size, terrain)

        @lru_cache(256)
        def render_terrain_army(
            self, size: int, terrain: int, army: int
        ) -> pygame.Surface:
            cell = self.render_terrain(size, terrain).copy()
            cell.blit(*self.render_army(size, army))
            return cell

        @lru_cache(16)
        def get_font(self, size: int) -> pygame.font.Font:
            return pygame.font.Font(self.font_, round((size - 12) ** 0.5) + 10)

        @lru_cache(512)
        def render_army(
            self, size: int, army: int
        ) -> tuple[pygame.Surface, pygame.Rect]:
            s = str(army)
            render = self.get_font(size).render
            text = render(s, True, WHITE)
            shadow = render(s, True, BLACK)
            surf = pygame.transform.box_blur(shadow, 2)
            surf.blit(text, (0, 0))
            m = size // 2
            rect = surf.get_rect(center=(m, m))
            return surf, rect

        @lru_cache(256)
        def render_terrain(self, size: int, terrain: int) -> pygame.Surface:
            if terrain < 0:
                if terrain < -2:
                    cell = self.render_index(size, 1)
                    if terrain == -4:
                        cell.blit(*self.render_image(size, self.obstacle))
                elif terrain < -1:
                    cell = self.render_index(size, 3)
                    cell.blit(*self.render_image(size, self.mountain))
                else:
                    cell = self.render_index(size, 4)
            elif terrain < 36:
                q, r = divmod(terrain, 12)
                cell = self.render_index(size, r + 5)
                if q == 1:
                    cell.blit(*self.render_image(size, self.city))
                elif q == 2:
                    cell.blit(*self.render_image(size, self.crown))
            else:
                cell = self.render_index(size, 2)
                if terrain == 37:
                    cell.blit(*self.render_image(size, self.city))
            return cell

        @classmethod
        def render_index(cls, size: int, index: int) -> pygame.Surface:
            cell = pygame.Surface((size, size))
            if index == 1:
                cell.fill(PALETTE[1])
            else:
                s = size - 1
                pygame.draw.rect(cell, PALETTE[index], (1, 1, s, s))
                pygame.draw.rect(cell, BLACK, (0, 0, size, size), 1)
            return cell

        @classmethod
        def render_image(
            cls, size: int, image: pygame.Surface
        ) -> tuple[pygame.Surface, pygame.Rect]:
            s = size * 25 // 32
            m = size // 2
            surf = pygame.transform.smoothscale(image, (s, s))
            rect = surf.get_rect(center=(m, m))
            return surf, rect

        @classmethod
        def all_cache_info(cls):
            return {
                "font": cls.get_font.cache_info(),
                "army": cls.render_army.cache_info(),
                "terrain": cls.render_terrain.cache_info(),
                "all": cls.render_terrain_army.cache_info(),
            }

    class BackgroundSprite(pygame.sprite.DirtySprite):
        def __init__(self, *group) -> None:
            super().__init__(*group)
            self.image = pygame.Surface((0, 0))
            self.rect = self.image.get_rect()
            self.layer = 0

        def fast_update(
            self,
            window_resize: bool,
            window_size: tuple[int, int] | None = None,
        ) -> None:
            assert isinstance(self.rect, pygame.Rect)
            if window_resize:
                window_size = window_size or pygame.display.get_window_size()
                self.image = pygame.Surface(window_size)
                self.rect.size = window_size
                self.image.fill(PALETTE[0])
                self.dirty = 1

    class MapSprite(pygame.sprite.DirtySprite):
        def __init__(
            self,
            *groups,
            window_size: tuple[int, int] | None = None,
        ) -> None:
            super().__init__(*groups)
            x, y = window_size or pygame.display.get_window_size()
            self.zoom = 3
            self.cell_size = 12
            self.center = pygame.Vector2(x // 2, y // 2)
            self.image = pygame.Surface((0, 0))
            self.rect = self.image.get_rect(center=self.center)
            self.data = None
            self.layer = 1
            self.resource = CachedResource()

            self.map_: list[int] = [0, 0]
            self.cities: list[int] = []

            # self._turn = 0

        @lru_cache(16)
        def get_image(self, size: tuple[int, int]) -> pygame.Surface:
            return pygame.Surface(size)

        def fast_update(
            self,
            zoom: int,
            move: tuple[float, float] | pygame.Vector2,
            data: GameUpdateDict | None,
        ) -> None:
            assert isinstance(self.rect, pygame.Rect)
            assert isinstance(self.image, pygame.Surface)

            skip = True
            resize = False

            if data:
                # if self._turn + 1 < data.turn:
                #     print(f"missing turn {self._turn + 1} to {data.turn - 1}")
                # self._turn = data.turn

                shape = self.map_[:2]
                self.map_ = patch(self.map_, data["map_diff"])
                self.cities = patch(self.cities, data["cities_diff"])
                if self.map_[:2] != shape:
                    resize = True

                skip = False
                self.data = data

            if not self.data:
                return

            if zoom and self.zoom != (z := max(0, min(self.zoom + zoom, 15))):
                skip = False
                resize = True
                s = 12 + z * z
                p = pygame.mouse.get_pos()
                c = self.center
                c -= p
                c *= s / self.cell_size
                c += p
                self.zoom = z
                self.cell_size = s

            if move and move != (0, 0):
                skip = False
                self.center += move
                self.rect.center = self.center

            if skip:
                return

            if resize:
                data = self.data
                c, r = self.map_[:2]
                c_size = self.cell_size
                size = c * c_size, r * c_size
                rect = self.rect
                rect.size = size
                rect.center = self.center
                self.image = self.get_image(size)

            if data:
                img = self.image
                res = self.resource
                c_size = self.cell_size

                c, r = self.rect.size
                yx = tuple(product(range(0, r, c_size), range(0, c, c_size)))

                c, r = self.map_[:2]
                n = c * r
                armies = self.map_[2 : 2 + n]
                terrain = self.map_[2 + n :]

                for i in self.cities:
                    if terrain[i] >= 0:
                        terrain[i] += 12
                    elif armies[i]:
                        terrain[i] = 37

                for i in data["generals"]:
                    if i >= 0 and terrain[i] >= 0:
                        terrain[i] += 24

                rr = self.rect.clip((0, 0), pygame.display.get_window_size())
                x, y = self.rect.topleft
                rr.move_ip(-x, -y)

                left = rr.left - self.cell_size
                right = rr.right
                top = rr.top - self.cell_size
                bottom = rr.bottom

                for (y, x), t, a in zip(yx, terrain, armies):
                    if not (left <= x < right and top <= y < bottom):
                        continue
                    if t == -1 and a:
                        t = 36
                    surf = res.render_cell(c_size, t, a)
                    img.blit(surf, (x, y))

            self.dirty = 1

    clock = pygame.time.Clock()

    bg = BackgroundSprite()
    bg.fast_update(True)
    map_ = MapSprite()

    group = pygame.sprite.LayeredDirty(bg, map_)
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


def lobby(window: pygame.Surface, queue: Queue[Any]) -> State | None:
    import pygame

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return

            match get_data(queue):
                case None:
                    continue
                case ("game_start", _data):
                    return game


def state_machine(queue: Queue[Any]) -> None:
    import pygame

    pygame.init()
    window = pygame.display.set_mode((800, 600))
    state: State | None = lobby
    while state is not None:
        state = state(window, queue)
    pygame.quit()


class PygameGUI:
    def __init__(self) -> None:
        self.queue: Queue[Any] = Queue()
        self.process = Process(target=state_machine, args=(self.queue,))

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


if __name__ == "__main__":
    gui = PygameGUI()
    gui.connect()
    gui.disconnect()
