"""Bounded Responses SSE assembly: partial output can never reach the overlay."""

import json

from dantetl.i18n import tr
from dantetl.translation.backend import RequestRejected
from dantetl.translation.limits import MAX_ENGLISH_CHARACTERS


class ResponseStream:
    def __init__(self) -> None:
        self._buffer = b""
        self._received = 0
        self.completed: dict | None = None

    def feed(self, data: bytes) -> None:
        self._received += len(data)
        if self._received > 131_072:
            raise RequestRejected(tr("ChatGPT response exceeded the size limit."))
        self._buffer += data
        # SSE accepts LF or CRLF. Bound the entire buffer before parsing it.
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            payload = b"\n".join(
                line[5:].lstrip() for line in event.split(b"\n") if line.startswith(b"data:")
            )
            if not payload or payload == b"[DONE]":
                continue
            try:
                value = json.loads(payload)
            except ValueError as error:
                raise RequestRejected(tr("ChatGPT returned an invalid response stream.")) from error
            if not isinstance(value, dict):
                raise RequestRejected(tr("ChatGPT returned an invalid response stream."))
            kind = value.get("type")
            if kind in ("error", "response.failed", "response.incomplete"):
                response = value.get("response", {})
                error = response.get("error", {}) if isinstance(response, dict) else {}
                code = error.get("code", value.get("code", "")) if isinstance(error, dict) else ""
                if code in (
                    "subscription_sharing_usage_limit_exceeded",
                    "subscription_sharing_usage_unavailable",
                ):
                    raise PlanUnavailable(
                        tr(
                            "ChatGPT plan usage is unavailable or at its limit. Manage "
                            "usage in Models."
                        )
                    )
                raise RequestRejected(
                    tr("ChatGPT did not complete this translation. No partial answer was shown.")
                )
            if kind == "response.completed":
                self.completed = value.get("response")

    def finish(self) -> tuple[str, object]:
        response = self.completed
        if not isinstance(response, dict) or response.get("status") != "completed":
            raise RequestRejected(tr("ChatGPT stream ended before a complete answer arrived."))
        output = response.get("output")
        if not isinstance(output, list) or any(
            not isinstance(item, dict) or item.get("type") not in ("message", "reasoning")
            for item in output
        ):
            raise RequestRejected(tr("ChatGPT returned no complete translation."))
        messages = [
            item for item in output if isinstance(item, dict) and item.get("type") == "message"
        ]
        if len(messages) != 1 or messages[0].get("role") != "assistant":
            raise RequestRejected(tr("ChatGPT returned an unsupported answer."))
        content = messages[0].get("content")
        if (
            not isinstance(content, list)
            or not content
            or any(
                not isinstance(item, dict)
                or item.get("type") != "output_text"
                or not isinstance(item.get("text"), str)
                for item in content
            )
        ):
            raise RequestRejected(tr("ChatGPT refused or did not complete this translation."))
        text = "".join(item["text"] for item in content).strip()
        if not text or len(text) > MAX_ENGLISH_CHARACTERS or "<think" in text.lower():
            raise RequestRejected(tr("ChatGPT returned empty, oversized or reasoning text."))
        return text, response.get("usage")


class PlanUnavailable(RequestRejected):
    pass
