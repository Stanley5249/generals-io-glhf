__all__ = (
    "patch",
    "make_diff",
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
