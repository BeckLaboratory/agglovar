"""String utility functions."""

from collections.abc import Container, Iterable, Iterator
from typing import Optional

def collision_rename(
        var_name: str,
        separator: Optional[str] = '.',
        *args: Container[str],
) -> str:
    """Rename a variable to avoid collisions with existing values.

    Adds "_n" to a variable name where "n" is the lowest integer that does not collide with an existing name. If
    `var_name` does not collide, it is returned unmodified.

    :param var_name: Variable name.
    :param separator: Separator to use between the variable name and the counter. If None, defaults to ".".
    :param args: One or more containers to search for matches.

    :return: Renamed string.
    """

    new_var_name = var_name
    i = 0

    separator = str(separator) if separator is not None else '.'

    while any([new_var_name in c for c in args]):
        i += 1
        new_var_name = f"{var_name}{separator}{i}"

    return new_var_name

def collision_rename_all(
        var_names: Iterable[str],
        separator: Optional[str] = '.',
        *args: Container[str],
) -> Iterator[str]:
    """Iterate through a list of names and deduplicate.

    :param var_names: Iterator of names.
    :param separator: Separator to use between the variable name and the counter. If None, defaults to ".".
    :param args: One or more containers to search for matches.

    :return: An iterator of deduplicated names.
    """
    var_name_set = set()

    args = [var_name_set] + list(args)

    for var_name in var_names:
        var_name = collision_rename(var_name, separator, *args)

        var_name_set.add(var_name)

        yield var_name

