"""Entry point for `python -m helios`."""

from helios import __version__


def main() -> None:
    print(f"Helios v{__version__}")


if __name__ == "__main__":
    main()
