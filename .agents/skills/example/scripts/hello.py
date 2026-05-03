from __future__ import annotations

import sys


def main() -> None:
    message = " ".join(sys.argv[1:]) or "hello from the example skill script"
    print(f"example skill script received: {message}")


if __name__ == "__main__":
    main()
