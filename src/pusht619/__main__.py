"""CLI entry point: ``python -m pusht619``."""

from . import __version__


def main() -> None:
    print(f"pusht619 v{__version__}")


if __name__ == "__main__":
    main()
