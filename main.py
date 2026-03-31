import sys

try:
    from woofalytics.app import main
except ModuleNotFoundError as exc:
    missing = exc.name or "dependency"
    print(
        f"Woofalytics is missing a required module: {missing}. "
        "Run `python scripts/check_setup.py` and install the reported prerequisites.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
