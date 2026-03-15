import pytest
import requests
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_proactive_self_emit_flows_through_openai_response_routes():
    global server
    server.api_surface = "openai"
    server.extra_env = {
        "VICUNA_SELF_EMIT_STARTUP_TEXT": "I have a proactive update for you.",
    }
    server.start()

    live_events = server.collect_named_sse_events(
        "GET",
        "/v1/responses/stream?after=0",
        stop_when=lambda event_name, _: event_name == "response.completed",
    )
    event_names = [event_name for event_name, _ in live_events]
    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]

    response_id = live_events[-1][1]["response"]["id"]
    response = server.make_request("GET", f"/v1/responses/{response_id}")
    assert response.status_code == 200
    assert response.body["id"] == response_id
    assert response.body["object"] == "response"
    assert response.body["output"][0]["role"] == "assistant"
    assert response.body["output"][0]["content"][0]["text"] == "I have a proactive update for you."

    replay_events = server.collect_named_sse_events(
        "GET",
        f"/v1/responses/{response_id}?stream=true",
        stop_when=lambda event_name, _: event_name == "response.completed",
    )
    assert [event_name for event_name, _ in replay_events] == event_names
    assert replay_events[-1][1]["response"]["id"] == response_id
    assert replay_events[4][1]["delta"] == "I have a proactive update for you."


def test_proactive_self_emit_stream_allows_only_one_live_client():
    global server
    server.api_surface = "openai"
    server.start()

    url = f"http://{server.server_host}:{server.server_port}/v1/responses/stream"
    stream_response = requests.get(url, stream=True)
    assert stream_response.status_code == 200
    try:
        blocked = server.make_request("GET", "/v1/responses/stream")
        assert blocked.status_code == 409
        assert blocked.body["error"]["type"] == "conflict_error"
    finally:
        stream_response.close()
