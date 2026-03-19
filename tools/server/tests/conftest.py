import os

import pytest
from utils import *


@pytest.fixture(scope="session", autouse=True)
def disable_active_loop_for_server_tests():
    previous = os.environ.get("VICUNA_ACTIVE_LOOP_ENABLED")
    os.environ["VICUNA_ACTIVE_LOOP_ENABLED"] = "0"
    yield
    if previous is None:
        os.environ.pop("VICUNA_ACTIVE_LOOP_ENABLED", None)
    else:
        os.environ["VICUNA_ACTIVE_LOOP_ENABLED"] = previous


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()
