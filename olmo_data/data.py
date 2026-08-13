"""Small compatibility layer for optional package data.

The upstream source snapshot referenced ``olmo_data`` but did not include the
package itself. Most paper configurations use remote paths and never enter
this code; keeping the helper here makes source installs and local tests work.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


def is_data_file(relative_path: str) -> bool:
    resource = files("olmo_data").joinpath(relative_path)
    return resource.is_file()


@contextmanager
def get_data_path(relative_path: str) -> Iterator[Path]:
    resource = files("olmo_data").joinpath(relative_path)
    with as_file(resource) as path:
        yield path
