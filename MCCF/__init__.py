"""MCCF – Vietnam power infrastructure mapping & analysis toolkit."""

from importlib import metadata

try:
    __version__ = metadata.version("mccf")
except metadata.PackageNotFoundError:  # pragma: no cover
    # Package is not installed.
    __version__ = "0.0.0"

__all__: list[str] = [
    "PDP8",
] 