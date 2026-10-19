"""The node's structured logger, with a stdlib shim for tests and non-node contexts."""

from __future__ import annotations

from typing import Any

try:
    from jarvis_log_client import JarvisLogger
except ImportError:  # unit tests / non-node contexts
    import logging

    class JarvisLogger:  # type: ignore[no-redef]  # minimal shim accepting structured kwargs
        def __init__(self, **kw: Any) -> None:
            self._log = logging.getLogger(kw.get("service", __name__))

        def info(self, msg: str, **kw: Any) -> None:
            self._log.info("%s %s", msg, kw or "")

        def warning(self, msg: str, **kw: Any) -> None:
            self._log.warning("%s %s", msg, kw or "")

        def error(self, msg: str, **kw: Any) -> None:
            self._log.error("%s %s", msg, kw or "")

        def debug(self, msg: str, **kw: Any) -> None:
            self._log.debug("%s %s", msg, kw or "")


logger = JarvisLogger(service="jarvis-node")
