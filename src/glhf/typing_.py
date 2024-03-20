from typing import Any, Literal, NamedTuple, TypedDict

__all__ = "QueueUpdateDict", "GameStartDict", "GameUpdateDict", "GameUpdateTuple"


class QueueUpdateDict(TypedDict):
    playerIndices: list[int]
    playerColors: list[int]
    lobbyIndex: int
    isForcing: bool
    numPlayers: int
    numForce: list[int]
    teams: list[int]
    usernames: list[None] | list[str]
    options: dict[str, Any]


class GameStartDict(TypedDict):
    playerIndex: int
    playerColors: list[int]
    replay_id: str
    chat_room: str
    usernames: list[str]
    teams: list[int]
    game_type: Literal["ffa", "1v1", "custom"]
    swamps: list
    lights: list
    options: list


class GameUpdateDict(TypedDict):
    scores: list[dict]
    turn: int
    stars: list[int]
    attackIndex: int
    generals: list[int]
    map_diff: list[int]
    cities_diff: list[int]


class GameUpdateTuple(NamedTuple):
    turn: int
    map_: list[int]
    cities: list[int]
    generals: list[int]
