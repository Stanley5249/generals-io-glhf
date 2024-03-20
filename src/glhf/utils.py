from __future__ import annotations

from asyncio import Event as AEvent
from asyncio import Queue as AQueue
from asyncio import Task, create_task
from collections import deque
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, Concatenate, Coroutine, Protocol, Self, overload

__all__ = (
    "to_task",
    "to_coro",
    "methodlike",
    "astreamify",
    "asignalize",
)


type MethodType[T, **P, R] = Callable[Concatenate[T, P], R]


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


def methodlike[T, **P, R](m: MethodType[T, P, R]) -> _MethodProtocol[T, P, R]:
    return m


def _put_task[R](coro: Coroutine[Any, Any, R], tasks: set[Task[R]]) -> None:
    task = create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.remove)


class _MethodProtocol[T, **P, R](Protocol):
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


class _MethodLike[T, **P, R](_MethodProtocol[T, P, R]):
    def __set_name__(self, owner: type[T], name: str) -> None:
        self.name = "_" + name


class _WrappedMethod[T, **P, R](_MethodLike[T, P, R]):
    def __init__(self, wrapped: MethodType[T, P, R]) -> None:
        self.wrapped = wrapped


class _AStream[**P, R]:
    __wrapped__: Callable[P, R]

    def __init__(self, wrapped: Callable[P, R]) -> None:
        update_wrapper(self, wrapped)
        self._queue: AQueue[R | None] = AQueue()
        self._tasks: set[Task[Any]] = set()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        item = self.__wrapped__(*args, **kwargs)
        _put_task(self._queue.put(item), self._tasks)
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

    def close(self) -> None:
        _put_task(self._queue.put(None), self._tasks)


class astreamify[T, **P, R](_WrappedMethod[T, P, R]):
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


class asignalize[T, **P, R](_WrappedMethod[T, P, R]):
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


class _Queue[R](deque[R]):
    def __call__(self, item: R) -> R:
        self.append(item)
        return item

    def get(self) -> R | None:
        try:
            return self.popleft()
        except IndexError:
            return None


class queue_method[T, R](_MethodLike[T, [R], R]):
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


class event_method[T](_MethodLike[T, [], None]):
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
