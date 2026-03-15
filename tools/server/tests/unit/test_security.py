import pytest
from openai import OpenAI
from utils import *

server = ServerPreset.tinyllama2()

TEST_API_KEY = "sk-this-is-the-secret-key"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.api_key = TEST_API_KEY


@pytest.mark.parametrize("endpoint", ["/health", "/models"])
def test_access_public_endpoint(endpoint: str):
    global server
    server.start()
    res = server.make_request("GET", endpoint)
    assert res.status_code == 200
    assert "error" not in res.body


@pytest.mark.parametrize("api_key", [None, "invalid-key"])
def test_incorrect_api_key(api_key: str):
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": f"Bearer {api_key}" if api_key else None,
    })
    assert res.status_code == 401
    assert "error" in res.body
    assert res.body["error"]["type"] == "authentication_error"


def test_correct_api_key():
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": f"Bearer {TEST_API_KEY}",
    })
    assert res.status_code == 200
    assert "error" not in res.body
    assert "content" in res.body


def test_correct_api_key_anthropic_header():
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "X-Api-Key": TEST_API_KEY,
    })
    assert res.status_code == 200
    assert "error" not in res.body
    assert "content" in res.body


def test_openai_surface_only_exposes_canonical_routes():
    global server
    server.api_surface = "openai"
    server.start()

    headers = {
        "Authorization": f"Bearer {TEST_API_KEY}",
    }

    for path in ["/", "/completion", "/chat/completions", "/api/chat", "/apply-template", "/tokenize", "/slots", "/models"]:
        res = server.make_request("GET" if path in {"/", "/slots", "/models"} else "POST", path, data={}, headers=headers)
        assert res.status_code == 404

    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    assert "error" not in res.body


def test_openai_surface_requires_bearer_auth_for_models_and_chat():
    global server
    server.api_surface = "openai"
    server.start()

    res = server.make_request("GET", "/v1/models")
    assert res.status_code == 401
    assert res.body["error"]["type"] == "authentication_error"

    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias,
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    })
    assert res.status_code == 401
    assert res.body["error"]["type"] == "authentication_error"

    res = server.make_request("GET", "/v1/models", headers={
        "Authorization": f"Bearer {TEST_API_KEY}",
    })
    assert res.status_code == 200
    assert "error" not in res.body


def test_openai_surface_rejects_non_bearer_auth_headers():
    global server
    server.api_surface = "openai"
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias,
        "messages": [
            {"role": "user", "content": "ping"},
        ],
    }, headers={
        "X-Api-Key": TEST_API_KEY,
    })
    assert res.status_code == 401
    assert res.body["error"]["type"] == "authentication_error"


def test_openai_surface_sets_request_id_headers():
    global server
    server.api_surface = "openai"
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "model": server.model_alias,
        "messages": [
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 8,
    }, headers={
        "Authorization": f"Bearer {TEST_API_KEY}",
        "X-Client-Request-Id": "client.trace.123",
    })
    assert res.status_code == 200
    assert res.headers["x-request-id"].startswith("req_")
    assert res.headers["x-client-request-id"] == "client.trace.123"


def test_openai_library_correct_api_key():
    global server
    server.start()
    client = OpenAI(api_key=TEST_API_KEY, base_url=f"http://{server.server_host}:{server.server_port}")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    )
    assert len(res.choices) == 1


@pytest.mark.parametrize("origin,cors_header,cors_header_value", [
    ("localhost", "Access-Control-Allow-Origin", "localhost"),
    ("web.mydomain.fr", "Access-Control-Allow-Origin", "web.mydomain.fr"),
    ("origin", "Access-Control-Allow-Credentials", "true"),
    ("web.mydomain.fr", "Access-Control-Allow-Methods", "GET, POST"),
    ("web.mydomain.fr", "Access-Control-Allow-Headers", "*"),
])
def test_cors_options(origin: str, cors_header: str, cors_header_value: str):
    global server
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert cors_header in res.headers
    assert res.headers[cors_header] == cors_header_value


@pytest.mark.parametrize(
    "media_path, image_url, success",
    [
        (None,             "file://mtmd/test-1.jpeg",    False), # disabled media path, should fail
        ("../../../tools", "file://mtmd/test-1.jpeg",    True),
        ("../../../tools", "file:////mtmd//test-1.jpeg", True),  # should be the same file as above
        ("../../../tools", "file://mtmd/notfound.jpeg",  False), # non-existent file
        ("../../../tools", "file://../mtmd/test-1.jpeg", False), # no directory traversal
    ]
)
def test_local_media_file(media_path, image_url, success,):
    server = ServerPreset.tinygemma3()
    server.media_path = media_path
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "test"},
                {"type": "image_url", "image_url": {
                    "url": image_url,
                }},
            ]},
        ],
    })
    if success:
        assert res.status_code == 200
    else:
        assert res.status_code == 400
