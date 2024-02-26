from __future__ import annotations

from asyncio import Event as _AsyncioEvent
from asyncio import Queue as _AsyncioQueue
from asyncio import Task, create_task
from collections import deque
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, Concatenate, Coroutine, Protocol, Self, overload

__all__ = (
    "patch",
    "make_diff",
    "to_task",
    "to_coro",
    "asyncio_queueify",
    "asyncio_eventify",
    "queue_method",
    "event_method",
)


def coord_to_name(coord: tuple[int, int]) -> str:
    """Converts a coordinate to a string representation."""
    return f"{coord[0]},{coord[1]}"


def coord_to_index(coord: tuple[int, int], col: int) -> int:
    """Converts a coordinate to an index in a 1D list representation."""
    return coord[0] * col + coord[1]


def patch(old: list[int], diff: list[int]) -> list[int]:
    new = []
    i = 0
    len_diff = len(diff)
    while i < len_diff:
        n = diff[i]
        if n:
            j = len(new)
            new += old[j : j + n]
        i += 1
        if i == len_diff:
            break
        n = diff[i]
        if n:
            j = i + 1
            new += diff[j : j + n]
            i += n
        i += 1
    return new


def make_diff(new: list[int], old: list[int]) -> list[int]:
    diff = []
    i = 0
    j = 0
    len_new = len(new)
    len_old = len(old)
    len_min = min(len_new, len_old)
    while True:
        while j < len_min and new[j] == old[j]:
            j += 1
        diff.append(j - i)
        i = j
        if j == len_min:
            if len_new > len_old:
                diff.append(len_new - len_old)
                diff += new[len_old:]
            break
        while j < len_min and new[j] != old[j]:
            j += 1
        if j < len_min:
            diff.append(j - i)
            diff += new[i:j]
            i = j
        else:
            diff.append(len_new - i)
            diff += new[i:len_new]
            break
    return diff


def to_task[**P, R](
    wrapped: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Task[R]]:
    @wraps(wrapped)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Task[R]:
        return create_task(wrapped(*args, **kwargs))

    return wrapper


def to_coro[**P, R](wrapped: Callable[P, R]) -> Callable[P, Coroutine[Any, Any, R]]:
    @wraps(wrapped)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return wrapped(*args, **kwargs)

    return wrapper


type MethodType[T, **P, R] = Callable[Concatenate[T, P], R]


class MethodLikeProtocol[T, **P, R](Protocol):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
        /,
    ) -> MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
        /,
    ) -> Callable[P, R]: ...


def methodlike[T, **P, R](m: MethodType[T, P, R]) -> MethodLikeProtocol[T, P, R]:
    return m


class MethodLike[T, **P, R](MethodLikeProtocol[T, P, R]):
    def __set_name__(self, owner: type[T], name: str) -> None:
        self.name = "_" + name


class _wrappedmethod[T, **P, R](MethodLike[T, P, R]):
    def __init__(self, wrapped: MethodType[T, P, R]) -> None:
        self.wrapped = wrapped


class AsyncioQueue[**P, R](_AsyncioQueue[R | None]):
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        super().__init__()
        update_wrapper(self, wrapped)
        self._tasks: set[Task] = set()
        self._close = _AsyncioEvent()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        item = self.__wrapped__(*args, **kwargs)
        task = create_task(self.put(item))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return item

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> R:
        item = await self.get()
        self.task_done()
        if item is None:
            raise StopAsyncIteration()
        return item

    def close(self) -> Task[None]:
        return create_task(self.put(None))


class asyncio_queueify[T, **P, R](_wrappedmethod[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> AsyncioQueue[P, R]: ...

    def __get__(
        self,
        instance: T | None,
        owner: type[T] | None = None,
    ) -> Any:
        if instance is None:
            return self.wrapped
        try:
            x = getattr(instance, self.name)
        except AttributeError:
            x = AsyncioQueue(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


class AsyncioEvent[**P, R](_AsyncioEvent):
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        super().__init__()
        update_wrapper(self, wrapped)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        res = self.__wrapped__(*args, **kwargs)
        self.set()
        return res


class asyncio_eventify[T, **P, R](_wrappedmethod[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> AsyncioEvent[P, R]: ...

    def __get__(
        self,
        instance: T | None,
        owner: type[T] | None = None,
    ) -> Any:
        if instance is None:
            return self.wrapped
        try:
            x = getattr(instance, self.name)
        except AttributeError:
            x = AsyncioEvent(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


class _Queue[R](deque[R]):
    def __call__(self, item: R) -> R:
        self.append(item)
        return item

    def get(self) -> R | None:
        try:
            return self.popleft()
        except IndexError:
            return None


class queue_method[T, R](MethodLike[T, [R], R]):
    def __call__(self, instance: T, item: R) -> R:
        return self.__get__(instance)(item)

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> Self: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _Queue[R]: ...

    def __get__(
        self,
        instance: T | None,
        owner: type[T] | None = None,
    ) -> Any:
        if instance is None:
            return self
        try:
            x = getattr(instance, self.name)
        except AttributeError:
            x = _Queue()
            setattr(instance, self.name, x)
        return x


class _Event:
    def __init__(self) -> None:
        self._set = False

    def __call__(self) -> None:
        self.set()

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    def wait(self) -> bool:
        x = self._set
        if self._set:
            self.clear()
        return x


class event_method[T](MethodLike[T, [], None]):
    def __call__(self, instance: T) -> None:
        return self.__get__(instance)()

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> Self: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _Event: ...

    def __get__(
        self,
        instance: T | None,
        owner: type[T] | None = None,
    ) -> Any:
        if instance is None:
            return self
        try:
            x = getattr(instance, self.name)
        except AttributeError:
            x = _Event()
            setattr(instance, self.name, x)
        return x


class Map(list[int]):
    @property
    def shape(self) -> list[int]:
        return self[:2:-1]

    @property
    def armies(self) -> list[list[int]]:
        m = self
        c, r = m[:2]
        stop = 2 + c * r
        return [m[i : i + r] for i in range(2, stop, r)]

    @property
    def terrain(self) -> list[list[int]]:
        m = self
        c, r = m[:2]
        n = c * r
        start = 2 + n
        return [m[i : i + r] for i in range(start, start + n, r)]


class Cities(list[int]):
    @property
    def cities(self) -> list[int]:
        return self.copy()
