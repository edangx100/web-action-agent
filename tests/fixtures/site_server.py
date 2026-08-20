"""HTTP server for the local static fixture site.

The browser agent needs somewhere to act that behaves like the real web and
costs nothing: real navigation, real form submission, real page loads, but no
network, no third-party host, and no state that survives a test. A plain static
file server over ``tests/fixtures/site/`` gives exactly that, and the operating
system picks the port so that parallel test sessions never collide.

Typical use is through the session-scoped ``fixture_site`` pytest fixture in
``tests/conftest.py``, but the class works standalone too::

    with StaticSite() as site:
        page.goto(site.url_for("search.html"))
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
"""Path: Directory holding the fixture site's HTML, CSS, and JavaScript."""


class _Handler(SimpleHTTPRequestHandler):
    """A quiet, non-caching variant of the standard library's file handler.

    Two behaviours of the default handler are unhelpful during a test run.
    It prints a line per request, which buries the actual test output, and it
    permits browser caching, which makes an agent's step-by-step trace hard to
    trust because a page may not have been re-fetched at all.

    Attributes:
        protocol_version: HTTP/1.1, so a browser may reuse one connection for a
            page and its assets instead of reconnecting for each.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Discard the per-request access log.

        Args:
            format: Printf-style format string supplied by the base class.
            *args: Values interpolated into ``format``.
        """

    def end_headers(self) -> None:
        """Append a no-caching header, then emit the response headers."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class StaticSite:
    """A static file server running on a background thread.

    The server binds to loopback on an ephemeral port, so several instances can
    run at once without coordination. Start it with :meth:`start` or by using
    the instance as a context manager, then address pages through
    :attr:`base_url` and :meth:`url_for`.

    Attributes:
        root: Directory served as the site's document root.

    Example:
        >>> with StaticSite() as site:
        ...     site.url_for("search.html")  # doctest: +SKIP
        'http://127.0.0.1:41234/search.html'
    """

    def __init__(self, root: Path | str = SITE_ROOT) -> None:
        """Prepare a server without starting it.

        Args:
            root: Directory to serve. Defaults to the fixture site shipped
                alongside this module.
        """
        self.root = Path(root)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        """Bind a port and begin serving on a background thread.

        Returns:
            This instance, so the call can be chained or used inline.

        Raises:
            RuntimeError: If this instance is already serving.
            FileNotFoundError: If :attr:`root` is not an existing directory,
                which usually means the fixture site was moved or renamed.
        """
        if self._server is not None:
            raise RuntimeError("this fixture site is already running")
        if not self.root.is_dir():
            raise FileNotFoundError(f"fixture site root does not exist: {self.root}")

        # The handler is constructed per request, so the document root has to be
        # bound in advance rather than passed at call time.
        handler = functools.partial(_Handler, directory=str(self.root))
        # Port 0 asks the operating system for any free port.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="fixture-site",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving, close the socket, and wait for the thread to finish.

        Safe to call more than once, and safe to call on a server that was
        never started; either way the instance ends up idle and restartable.
        """
        if self._server is None:
            return

        self._server.shutdown()  # Breaks out of serve_forever.
        self._server.server_close()  # Releases the port for the next session.
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        """int: The TCP port the operating system assigned at start-up.

        Raises:
            RuntimeError: If the server is not currently running, in which case
                no port has been assigned yet.
        """
        if self._server is None:
            raise RuntimeError("the fixture site is not running")
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        """str: Root URL of the running site, with a trailing slash."""
        return f"http://127.0.0.1:{self.port}/"

    def url_for(self, path: str) -> str:
        """Build an absolute URL for a page of the site.

        Args:
            path: Site-relative path, with or without a leading slash, such as
                ``"search.html"`` or ``"static/site.css"``. A query string may
                be included: ``"results.html?q=archive"``.

        Returns:
            The absolute URL, for example
            ``"http://127.0.0.1:41234/search.html"``.
        """
        return urljoin(self.base_url, path.lstrip("/"))

    def __enter__(self) -> Self:
        """Start the server on entering a ``with`` block.

        Returns:
            This instance, already serving.
        """
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the server on leaving a ``with`` block, error or not.

        Args:
            exc_type: Class of the exception that ended the block, if any.
            exc: The exception instance, if any.
            tb: The associated traceback, if any.
        """
        self.stop()
