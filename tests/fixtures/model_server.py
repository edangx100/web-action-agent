"""A stand-in AI model, served over HTTP on this machine, for runs with no key.

The agent talks to its model the way it talks to any hosted one: HTTP requests
in the OpenAI chat format, to an address set by its profile. The ``local``
profile points at ``127.0.0.1`` and needs no key, because it is meant for a
model server running on your own machine. This module is such a server, with
the model replaced by one fixed rule, so the whole evaluation -- browser,
agent, checks, results tables -- can run with no account, no key and no
network beyond this computer.

**The rule.** The practice site's benchmark tasks all ask for the library's
name, and every page of the site carries it in its title ("Search the
catalogue -- Northgate Public Library"). So the stand-in reads the page title
the agent sent it and answers with the part after the last dash. It is not
handed the answer: if the browser never loaded the page, there is no title to
read, and the stand-in reports that it failed. A pass therefore shows that the
page really was fetched from the local site, read by the agent, and sent here.

**What it is not.** It is not a model and measures nothing about one. It
exists to show that everything around the model works on a fresh machine.

Typical use::

    with StandInModel() as model:
        environment["WAA_BASE_URL"] = model.base_url
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self

#: The line of the agent's prompt that carries the page title, as the planner
#: writes it ("Title: Search the catalogue -- Northgate Public Library").
TITLE_LINE = re.compile(r"^Title: (.*)$", re.MULTILINE)

#: The separator between a page's own name and the site's name in a title:
#: an em dash, an en dash or a hyphen, with a space either side.
TITLE_SEPARATOR = re.compile(r"\s[—–-]\s")

#: Words that only the start-of-sweep connection test contains. That call
#: asks for ``{"ok": true}`` rather than an action.
CONNECTION_TEST = "connection test"


def prompt_text(request: dict[str, Any]) -> str:
    """Join the text of every message in a chat request, in order.

    A message's content is either a string or, when it carries a picture, a
    list of parts; only the text parts are kept.

    Args:
        request: The JSON body of a chat-completions request.

    Returns:
        str: All the text the model was sent.
    """
    texts: list[str] = []
    for message in request.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(part.get("text", "") for part in content if part.get("type") == "text")
    return "\n".join(texts)


def reply_to(request: dict[str, Any]) -> str:
    """Decide the stand-in's answer to one request.

    Args:
        request: The JSON body of a chat-completions request.

    Returns:
        str: The reply text, which is JSON: ``{"ok": true}`` for the
        connection test, otherwise a ``done`` action. A ``success`` with the
        site's name when the prompt shows a page title, and a ``failed`` with
        the reason when it does not.
    """
    text = prompt_text(request)
    if CONNECTION_TEST in text:
        return json.dumps({"ok": True})
    # The last title in the prompt is the page the browser is on now; earlier
    # ones, if any, would be from pages already left behind.
    titles = TITLE_LINE.findall(text)
    title = titles[-1].strip() if titles else ""
    if not title:
        action = {
            "action": "done",
            "status": "failed",
            "answer": None,
            "reason": "the page has no title to read the library's name from",
        }
    else:
        # "Search the catalogue -- Northgate Public Library" -> the part after
        # the last dash. A title with no dash is the name on its own.
        name = TITLE_SEPARATOR.split(title)[-1].strip()
        action = {"action": "done", "status": "success", "answer": name, "reason": None}
    return json.dumps(action)


def completion(request: dict[str, Any], content: str) -> dict[str, Any]:
    """Wrap a reply in the response shape an OpenAI-style service returns.

    Args:
        request: The request being answered, for the model name.
        content: The reply text.

    Returns:
        dict[str, Any]: The response body. Token counts are rough (four
        characters to a token), there so the agent's cost records have
        something to count.
    """
    return {
        "id": f"standin-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", "stand-in"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_text(request)) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (len(prompt_text(request)) + len(content)) // 4,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    """Answers ``POST .../chat/completions``; everything else is a 404."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Read one chat request, decide the reply, and send it back."""
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": f"no such endpoint: {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "the request body is not JSON"}})
            return
        self.server.calls.append(request)  # type: ignore[attr-defined]
        self._send(200, completion(request, reply_to(request)))

    def _send(self, status: int, body: dict[str, Any]) -> None:
        """Send a JSON response.

        Args:
            status: The HTTP status code.
            body: The response, to be sent as JSON.
        """
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Discard the per-request access log.

        Args:
            format: Printf-style format string supplied by the base class.
            *args: Values interpolated into ``format``.
        """


class StandInModel:
    """The stand-in model server, running on a background thread.

    Binds to loopback on a port the operating system picks, like the
    practice site's server, so it can never be reached from another machine
    and never collides with anything else running.

    Attributes:
        calls: Every request answered, as parsed JSON, in order.
    """

    def __init__(self) -> None:
        """Prepare a server without starting it."""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.calls: list[dict[str, Any]] = []

    def start(self) -> Self:
        """Bind a port and begin serving on a background thread.

        Returns:
            This instance, so the call can be chained.

        Raises:
            RuntimeError: If this instance is already serving.
        """
        if self._server is not None:
            raise RuntimeError("this stand-in model is already running")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        # The handler is built anew per request and can reach only the
        # server, so the shared list of calls lives there.
        self._server.calls = self.calls  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="stand-in-model", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and wait for the thread. Safe to call more than once."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        """str: The API root to give the agent, such as ``http://127.0.0.1:41235/v1``.

        Raises:
            RuntimeError: If the server is not running.
        """
        if self._server is None:
            raise RuntimeError("this stand-in model is not running")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> Self:
        """Start serving on entry to a ``with`` block."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop serving on exit from a ``with`` block, whatever happened."""
        self.stop()
