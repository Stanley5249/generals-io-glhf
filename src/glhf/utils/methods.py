from asyncio import Event as AEvent
from asyncio import Queue as AQueue
from asyncio import QueueEmpty, Task, create_task
from functools import partial, update_wrapper, wraps
from queue import Empty, Queue
from threading import Event
from typing import Any, Callable, Concatenate, Coroutine, Protocol, Self, overload

__all__ = [
    "methodlike",
    "streamify",
    "signalize",
    "astreamify",
    "asignalize",
    "to_coro",
    "to_task",
]


# ============================================================
# typing
# ============================================================


class MethodLike[T, **P, R](Protocol):
    @overload
    def __get__(
        self, instance: None, owner: type[T], /
    ) -> Callable[Concatenate[T, P], R]: ...

    @overload
    def __get__(
        self, instance: T, owner: type[T] | None = None, /
    ) -> Callable[P, R]: ...


class MethodDescriptor[T, **P, R](MethodLike[T, P, R]):
    wrapper: Callable[..., Callable[P, R]]

    def __init_subclass__(cls, /, wrapper: Callable[..., Callable[P, R]]) -> None:
        cls.wrapper = wrapper

    def __init__(self, wrapped: Callable[Concatenate[T, P], R]) -> None:
        self.wrapped = wrapped

    def __set_name__(self, owner: type[T], name: str) -> None:
        self.name = "_" + name

    def __get__(self, instance: T | None, owner: type[T] | None = None) -> Any:
        if instance is None:
            return self.wrapped
        try:
            x = getattr(instance, self.name)
        except AttributeError:
            x = self.wrapper(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


def methodlike[T, **P, R](m: Callable[Concatenate[T, P], R]) -> MethodLike[T, P, R]:
    return m


# ============================================================
# synchronous
# ============================================================


class Stream[**P, R]:
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        update_wrapper(self, wrapped)
        self._queue: Queue[R | None] = Queue()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        item = self.__wrapped__(*args, **kwargs)
        self._queue.put_nowait(item)
        return item

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> R:
        q = self._queue
        item = q.get()
        q.task_done()
        if item is None:
            raise StopIteration()
        return item

    def wait(self) -> R | None:
        q = self._queue
        item = q.get()
        q.task_done()
        return item

    def get(self) -> R | None:
        q = self._queue
        try:
            item = q.get_nowait()
        except Empty:
            return None
        else:
            q.task_done()
            return item

    def close(self) -> None:
        self._queue.put_nowait(None)


class Signal[**P, R]:
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        update_wrapper(self, wrapped)
        self.event = Event()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self.event.set()
        return self.__wrapped__(*args, **kwargs)

    def get(self) -> bool:
        s = self.event.is_set()
        if s:
            self.event.clear()
        return s

    def wait(self) -> None:
        self.event.wait()
        self.event.clear()


class streamify[T, **P, R](MethodDescriptor[T, P, R], wrapper=Stream[P, R]):
    pass


class signalize[T, **P, R](MethodDescriptor[T, P, R], wrapper=Signal[P, R]):
    pass


# ============================================================
# asynchronous
# ============================================================


class AStream[**P, R]:
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        update_wrapper(self, wrapped)
        self._queue: AQueue[R | None] = AQueue()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        item = self.__wrapped__(*args, **kwargs)
        self._queue.put_nowait(item)
        return item

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> R:
        q = self._queue
        item = await q.get()
        q.task_done()
        if item is None:
            raise StopAsyncIteration()
        return item

    async def wait(self) -> R | None:
        q = self._queue
        item = await q.get()
        q.task_done()
        return item

    def get(self) -> R | None:
        q = self._queue
        try:
            item = q.get_nowait()
        except QueueEmpty:
            return None
        else:
            q.task_done()
            return item

    def close(self) -> None:
        self._queue.put_nowait(None)


class ASignal[**P, R]:
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        update_wrapper(self, wrapped)
        self.event = AEvent()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self.event.set()
        return self.__wrapped__(*args, **kwargs)

    def get(self) -> bool:
        s = self.event.is_set()
        if s:
            self.event.clear()
        return s

    async def wait(self) -> None:
        await self.event.wait()
        self.event.clear()


class astreamify[T, **P, R](MethodDescriptor[T, P, R], wrapper=AStream[P, R]):
    pass


class asignalize[T, **P, R](MethodDescriptor[T, P, R], wrapper=ASignal[P, R]):
    pass


def to_coro[**P, R](wrapped: Callable[P, R]) -> Callable[P, Coroutine[Any, Any, R]]:
    @wraps(wrapped)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return wrapped(*args, **kwargs)

    return wrapper


def to_task[**P, R](
    wrapped: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Task[R]]:
    @wraps(wrapped)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Task[R]:
        return create_task(wrapped(*args, **kwargs))

    return wrapper
