"""OpenAI adapter: wraps the ``httpx`` transport, not the client surface.

One implementation catches Chat Completions, the Responses API, Assistants,
embeddings and the Agents SDK, and survives most SDK refactors because the
wire format is the stable contract. Usage::

    from swarmscope.adapters.openai_transport import instrument_openai
    instrument_openai(sdk)            # patches OpenAI/AsyncOpenAI constructors
    instrument_openai(sdk, client)    # or wraps one existing client

Streaming responses are handled by tee-ing the SSE body and recording usage
from the final chunk when ``stream_options={"include_usage": True}`` was
requested; otherwise the generation is recorded with zero tokens and
``attrs["stream_usage_missing"]=True`` so the gap is visible, not silent.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..core.sdk import Swarmscope

log = logging.getLogger("swarmscope.adapters.openai")

_USAGE_PATHS = ("/chat/completions", "/responses", "/completions", "/embeddings", "/messages")


def _parse_usage(body: dict[str, Any]) -> dict[str, Any] | None:
    u = body.get("usage") or {}
    if not u and "response" in body and isinstance(body["response"], dict):  # streamed responses API
        u = body["response"].get("usage") or {}
    if not u:
        return None
    details_in = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    details_out = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {
        "input_tokens": int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0),
        "output_tokens": int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0),
        "cached_input_tokens": int(details_in.get("cached_tokens", 0) or 0),
        "reasoning_tokens": int(details_out.get("reasoning_tokens", 0) or 0),
        "model": body.get("model") or (body.get("response") or {}).get("model"),
        "request_id": body.get("id"),
    }


def _record(sdk: Swarmscope, body: dict[str, Any], model_hint: str | None, t0: float, system: str = "openai",
            **attrs) -> None:
    u = _parse_usage(body)
    if u is None:
        sdk.generation(model=model_hint, latency_ms=(time.perf_counter() - t0) * 1000, system=system,
                       stream_usage_missing=True, **attrs)
        return
    model = u.pop("model", None) or model_hint
    sdk.generation(model=model, latency_ms=(time.perf_counter() - t0) * 1000, system=system, **u, **attrs)


def _sse_final_usage(text: str) -> dict[str, Any]:
    """Scan SSE chunks; return the last JSON object carrying a usage block."""
    last: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage") or (isinstance(obj.get("response"), dict) and obj["response"].get("usage")):
            last = obj
        elif not last and obj.get("model"):
            last = {"model": obj.get("model"), "id": obj.get("id")}
    return last


def _wants_tracking(request) -> tuple[bool, str | None]:
    path = request.url.path
    if not any(path.endswith(p) for p in _USAGE_PATHS):
        return False, None
    model = None
    try:
        body = json.loads(request.content or b"{}")
        model = body.get("model")
    except Exception:
        pass
    return True, model


class _TransportWrapper:
    """Sync httpx transport wrapper."""

    def __init__(self, inner, sdk: Swarmscope, system: str = "openai") -> None:
        self._inner, self._sdk, self._system = inner, sdk, system

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def handle_request(self, request):
        track, model = _wants_tracking(request)
        t0 = time.perf_counter()
        response = self._inner.handle_request(request)
        if not track:
            return response
        ctype = response.headers.get("content-type", "")
        try:
            if "text/event-stream" in ctype and hasattr(response, "_content"):
                # Body already loaded (e.g. mock transports): parse it directly.
                _record(self._sdk, _sse_final_usage(response.content.decode("utf-8", "ignore")), model, t0,
                        self._system, streamed=True)
            elif "text/event-stream" in ctype:
                import httpx

                original_iter = response.stream
                sdk, system = self._sdk, self._system

                class _Tee(httpx.SyncByteStream):
                    """The SDK breaks out at ``[DONE]`` and closes the stream, so the
                    generation is recorded on close (or at exhaustion, whichever first)."""

                    def __init__(self):
                        self._buf = bytearray()
                        self._done = False

                    def _finish(self):
                        if self._done:
                            return
                        self._done = True
                        try:
                            _record(sdk, _sse_final_usage(self._buf.decode("utf-8", "ignore")), model, t0, system,
                                    streamed=True)
                        except Exception:
                            log.exception("swarmscope: failed to record streamed generation")

                    def __iter__(self):
                        try:
                            for chunk in original_iter:
                                self._buf.extend(chunk)
                                yield chunk
                        finally:  # runs on exhaustion *and* when the consumer breaks early
                            self._finish()

                    def close(self):
                        self._finish()
                        original_iter.close()

                response.stream = _Tee()
            else:
                response.read()
                if response.status_code < 400:
                    _record(self._sdk, response.json(), model, t0, self._system)
                else:
                    self._sdk.generation(model=model, latency_ms=(time.perf_counter() - t0) * 1000,
                                         system=self._system, http_status=response.status_code, error=True)
        except Exception:
            log.exception("swarmscope: openai transport hook failed (request unaffected)")
        return response

    def close(self):
        self._inner.close()


class _AsyncTransportWrapper:
    def __init__(self, inner, sdk: Swarmscope, system: str = "openai") -> None:
        self._inner, self._sdk, self._system = inner, sdk, system

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def handle_async_request(self, request):
        track, model = _wants_tracking(request)
        t0 = time.perf_counter()
        response = await self._inner.handle_async_request(request)
        if not track:
            return response
        ctype = response.headers.get("content-type", "")
        try:
            if "text/event-stream" in ctype and hasattr(response, "_content"):
                _record(self._sdk, _sse_final_usage(response.content.decode("utf-8", "ignore")), model, t0,
                        self._system, streamed=True)
            elif "text/event-stream" in ctype:
                import httpx

                original = response.stream
                sdk, system = self._sdk, self._system

                class _ATee(httpx.AsyncByteStream):
                    def __init__(self):
                        self._buf = bytearray()
                        self._done = False

                    def _finish(self):
                        if self._done:
                            return
                        self._done = True
                        try:
                            _record(sdk, _sse_final_usage(self._buf.decode("utf-8", "ignore")), model, t0, system,
                                    streamed=True)
                        except Exception:
                            log.exception("swarmscope: failed to record streamed generation")

                    async def __aiter__(self):
                        try:
                            async for chunk in original:
                                self._buf.extend(chunk)
                                yield chunk
                        finally:
                            self._finish()

                    async def aclose(self):
                        self._finish()
                        await original.aclose()

                response.stream = _ATee()
            else:
                await response.aread()
                if response.status_code < 400:
                    _record(self._sdk, response.json(), model, t0, self._system)
                else:
                    self._sdk.generation(model=model, latency_ms=(time.perf_counter() - t0) * 1000,
                                         system=self._system, http_status=response.status_code, error=True)
        except Exception:
            log.exception("swarmscope: openai async transport hook failed (request unaffected)")
        return response

    async def aclose(self):
        await self._inner.aclose()


def wrap_client(sdk: Swarmscope, client, system: str = "openai"):
    """Wrap the transport of one ``OpenAI``/``AsyncOpenAI`` (or bare httpx) client in place."""
    http = getattr(client, "_client", client)  # openai.OpenAI -> httpx.Client
    transport = getattr(http, "_transport", None)
    if transport is None:
        raise TypeError("client has no httpx transport to wrap")
    if isinstance(transport, (_TransportWrapper, _AsyncTransportWrapper)):
        # Already wrapped: re-target to this SDK instance (a client may outlive the SDK it was first
        # wrapped with, e.g. one client shared across several runs/instances).
        transport._sdk = sdk
        return client
    import httpx

    if isinstance(http, httpx.AsyncClient):
        http._transport = _AsyncTransportWrapper(transport, sdk, system)
    else:
        http._transport = _TransportWrapper(transport, sdk, system)
    return client


_patched: dict[str, Any] = {}


def instrument_openai(sdk: Swarmscope, client=None, *, system: str = "openai"):
    """Instrument one client, or patch the ``openai`` constructors for all future clients."""
    if client is not None:
        return wrap_client(sdk, client, system)
    import openai  # noqa: F401 - requires the extra

    for cls_name in ("OpenAI", "AsyncOpenAI"):
        cls = getattr(openai, cls_name, None)
        if cls is None or cls_name in _patched:
            continue
        orig_init = cls.__init__

        def __init__(self, *a, _orig=orig_init, **kw):
            _orig(self, *a, **kw)
            try:
                wrap_client(sdk, self, system)
            except Exception:
                log.exception("swarmscope: could not wrap %s", type(self).__name__)

        cls.__init__ = __init__
        _patched[cls_name] = orig_init
    return None


def uninstrument_openai() -> None:
    import openai

    for cls_name, orig in list(_patched.items()):
        getattr(openai, cls_name).__init__ = orig
        _patched.pop(cls_name)
