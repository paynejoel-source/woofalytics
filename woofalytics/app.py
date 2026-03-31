from __future__ import annotations

import logging
import signal
import sys

from .config import AppConfig
from .service import BarkMonitor
from .web import build_server


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AppConfig()
    try:
        monitor = BarkMonitor(config)
        monitor.start()
    except RuntimeError as exc:
        logging.getLogger("Woofalytics").error("%s", exc)
        raise SystemExit(1)

    server = build_server(config.host, config.port, monitor)
    logger = logging.getLogger("Woofalytics")

    def handle_stop(signum, frame) -> None:  # noqa: ARG001
        logger.info("Stopping Woofalytics...")
        server.shutdown()
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    logger.info("Serving Woofalytics at http://%s:%s", config.host, config.port)
    try:
        server.serve_forever()
    finally:
        monitor.stop()
