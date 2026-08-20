"""Serves the static fixture site over HTTP on an ephemeral port.

The agent needs somewhere to act that behaves like the web and costs nothing:
real navigation, real form submission, real page loads, but no network, no
third-party host, and no state that survives a test. A plain static file server
over `tests/fixtures/site/` gives exactly that. The port is chosen by the
kernel so that parallel test sessions never collide.
"""

from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Self
from urllib.parse import urljoin

SITE_ROOT = Path(__file__).resolve().parent / "site"


class _Handler(SimpleHTTPRequestHandler):
    """A quiet, non-caching variant of the stdlib static handler.

    Request logging would bury the test output, and a browser that caches a
    page between steps makes an agent trace hard to read.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class StaticSite:
    """A running static file server, addressable by `base_url` / `url_for`."""

    def __init__(self, root: Path | str = SITE_ROOT) -> None:
        self.root = Path(root)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        if self._server is not None:
            raise RuntimeError("this fixture site is already running")
        if not self.root.is_dir():
            raise FileNotFoundError(f"fixture site root does not exist: {self.root}")

        handler = functools.partial(_Handler, directory=str(self.root))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fixture-site",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("the fixture site is not running")
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def url_for(self, path: str) -> str:
        """Absolute URL for a site-relative path such as `search.html`."""
        return urljoin(self.base_url, path.lstrip("/"))

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
