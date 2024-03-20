from asyncio import Event as _AsyncioEvent
from asyncio import Queue as _AsyncioQueue
from asyncio import Task, create_task
from collections import deque
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, Concatenate, Coroutine, Protocol, Self, overload

__all__ = (
    "to_task",
    "to_coro",
    "MethodLikeProtocol",
    "methodlike",
    "MethodLike",
    "AsyncioQueue",
    "asyncio_queueify",
    "AsyncioEvent",
    "asyncio_eventify",
    "queue_method",
    "event_method",
)


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
