"""``python -m optimizer.run`` — run the full optimize->verify->publish loop.

Thin alias to the framework's real entry point (``implementation/run_overnight.py``
-> ``overnight.main``), so a user has ONE discoverable command for the whole
north-star loop. All flags pass through unchanged (try ``--backend mock`` for a
hardware-free end-to-end demo in ~90s).
"""

from __future__ import annotations


def main() -> int:
    # overnight.main() parses sys.argv directly, so flags after
    # ``python -m optimizer.run`` pass through unchanged.
    from overnight import main as overnight_main  # noqa: PLC0415 (src on path via package __init__)
    ret = overnight_main()
    return int(ret) if isinstance(ret, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
