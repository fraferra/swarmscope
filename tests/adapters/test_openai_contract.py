"""Contract tests for the httpx-transport OpenAI adapter. No network: httpx.MockTransport."""
import json

import pytest

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")

import swarmscope as ss
from swarmscope.adapters.openai_transport import instrument_openai, uninstrument_openai, wrap_client

CHAT = {"id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4o-mini-2024-07-18", "created": 1,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
                  "prompt_tokens_details": {"cached_tokens": 4}}}
RESPONSES = {"id": "resp_1", "object": "response", "model": "gpt-4.1", "created_at": 1, "status": "completed",
             "output": [{"type": "message", "id": "m", "status": "completed", "role": "assistant",
                         "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
             "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                       "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 2}},
             "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}


LAZY = {"on": True}


class _LazyStream(httpx.SyncByteStream):
    """Yields the body in pieces without httpx pre-reading it (like a real socket)."""

    def __init__(self, data: bytes):
        self._data = data

    def __iter__(self):
        for i in range(0, len(self._data), 40):
            yield self._data[i:i + 40]


def _sse(chunks):
    body = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    headers = {"content-type": "text/event-stream"}
    if LAZY["on"]:
        return httpx.Response(200, stream=_LazyStream(body), headers=headers)
    return httpx.Response(200, content=body, headers=headers)  # httpx pre-reads bytes content


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    if request.url.path.endswith("/chat/completions"):
        if body.get("stream"):
            chunks = [{"id": "c", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1,
                       "choices": [{"index": 0, "delta": {"content": "h"}, "finish_reason": None}]},
                      {"id": "c", "object": "chat.completion.chunk", "model": "gpt-4o-mini", "created": 1,
                       "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}}]
            return _sse(chunks)
        return httpx.Response(200, json=CHAT)
    if request.url.path.endswith("/responses"):
        return httpx.Response(200, json=RESPONSES)
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"object": "list", "data": []})
    return httpx.Response(500, json={"error": "boom"})


@pytest.fixture
def sdk():
    s = ss.Swarmscope("memory://", flush_interval=0.01)
    yield s
    s.close()


def _client():
    return openai.OpenAI(api_key="test", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(_handler)))


def test_chat_completion_usage_attributed(sdk):
    client = wrap_client(sdk, _client())
    with sdk.run("oa") as run:
        with sdk.agent_scope(role="w") as a:
            r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
            assert r.choices[0].message.content == "hi"
    gens = sdk.store.events(run.run_id, types=["generation"])
    assert len(gens) == 1
    g = gens[0]
    assert g.agent_id == a and g.model == "gpt-4o-mini-2024-07-18" and g.system == "openai"
    assert (g.input_tokens, g.output_tokens, g.cached_input_tokens) == (12, 3, 4)
    assert g.cost_usd is not None and g.latency_ms is not None and g.request_id == "chatcmpl-1"


def test_responses_api_usage(sdk):
    client = wrap_client(sdk, _client())
    with sdk.run("oa") as run:
        client.responses.create(model="gpt-4.1", input="x")
    g = sdk.store.events(run.run_id, types=["generation"])[0]
    assert (g.input_tokens, g.output_tokens, g.reasoning_tokens) == (20, 5, 2) and g.model == "gpt-4.1"


@pytest.mark.parametrize("lazy", [True, False])
def test_streaming_usage_from_final_chunk(sdk, lazy):
    LAZY["on"] = lazy
    client = wrap_client(sdk, _client())
    with sdk.run("oa") as run:
        stream = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                                                stream=True, stream_options={"include_usage": True})
        text = "".join((c.choices[0].delta.content or "") for c in stream if c.choices)
        assert text == "h"
    sdk.flush()
    g = sdk.store.events(run.run_id, types=["generation"])[0]
    assert (g.input_tokens, g.output_tokens) == (7, 1) and g.attrs.get("streamed") is True


def test_non_tracked_paths_ignored_and_errors_recorded(sdk):
    client = wrap_client(sdk, _client())
    with sdk.run("oa") as run:
        client.models.list()
        with pytest.raises(openai.APIStatusError):
            client.embeddings.create(model="text-embedding-3-small", input="x")
    gens = sdk.store.events(run.run_id, types=["generation"])
    assert len(gens) == 1 and gens[0].attrs.get("error") is True and gens[0].attrs["http_status"] == 500


def test_global_instrumentation_patches_constructor(sdk):
    instrument_openai(sdk)
    try:
        client = _client()
        with sdk.run("oa") as run:
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert sdk.store.events(run.run_id, types=["generation"])
    finally:
        uninstrument_openai()


async def test_async_client(sdk):
    aclient = openai.AsyncOpenAI(api_key="test", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)))
    wrap_client(sdk, aclient)
    with sdk.run("oa") as run:
        with sdk.agent_scope() as a:
            await aclient.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
    g = sdk.store.events(run.run_id, types=["generation"])[0]
    assert g.agent_id == a and g.input_tokens == 12


def test_wrap_client_retargets_to_new_sdk():
    """One client shared across SDK instances: generations go to the SDK that wrapped it last."""
    client = _client()
    a = ss.Swarmscope("memory://", flush_interval=0.01)
    b = ss.Swarmscope("memory://", flush_interval=0.01)
    try:
        wrap_client(a, client)
        with a.run("a") as ra:
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        wrap_client(b, client)  # idempotent wrap, but re-targeted
        with b.run("b") as rb:
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])
        assert len(a.store.events(ra.run_id, types=["generation"])) == 1
        assert len(b.store.events(rb.run_id, types=["generation"])) == 1
        assert not a.store.events(rb.run_id, types=["generation"])
    finally:
        a.close()
        b.close()
