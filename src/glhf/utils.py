from asyncio import Event as AEvent
from asyncio import Queue as AQueue
from asyncio import QueueEmpty, Task, create_task
from functools import partial, update_wrapper, wraps
from queue import Empty, Queue
from threading import Event
from typing import Any, Callable, Concatenate, Coroutine, Protocol, Self, overload

__all__ = (
    "methodlike",
    "streamify",
    "signalize",
    "astreamify",
    "asignalize",
    "to_coro",
    "to_task",
)


# ============================================================
# typing
# ============================================================

type _MethodType[T, **P, R] = Callable[Concatenate[T, P], R]


class _MethodLike[T, **P, R](Protocol):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
        /,
    ) -> _MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
        /,
    ) -> Callable[P, R]: ...


class _MethodDescriptor[T, **P, R](_MethodLike[T, P, R]):
    def __set_name__(self, owner: type[T], name: str) -> None:
        self.name = "_" + name

    def __init__(self, wrapped: _MethodType[T, P, R]) -> None:
        self.wrapped = wrapped


def methodlike[T, **P, R](m: _MethodType[T, P, R]) -> _MethodLike[T, P, R]:
    return m


# ============================================================
# synchronous
# ============================================================


class _Stream[**P, R]:
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

    def wait(self) -> R:
        return self.__next__()

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


class _Signal[**P, R]:
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


class streamify[T, **P, R](_MethodDescriptor[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> _MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _Stream[P, R]: ...

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
            x = _Stream(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


class signalize[T, **P, R](_MethodDescriptor[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> _MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _Signal[P, R]: ...

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
            x = _Signal(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


# ============================================================
# asynchronous
# ============================================================


class _AStream[**P, R]:
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

    async def wait(self) -> R:
        return await self.__anext__()

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


class _ASignal[**P, R]:
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


class astreamify[T, **P, R](_MethodDescriptor[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> _MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _AStream[P, R]: ...

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
            x = _AStream(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


class asignalize[T, **P, R](_MethodDescriptor[T, P, R]):
    @overload
    def __get__(
        self,
        instance: None,
        owner: type[T],
    ) -> _MethodType[T, P, R]: ...

    @overload
    def __get__(
        self,
        instance: T,
        owner: type[T] | None = None,
    ) -> _ASignal[P, R]: ...

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
            x = _ASignal(partial(self.wrapped, instance))
            setattr(instance, self.name, x)
        return x


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
