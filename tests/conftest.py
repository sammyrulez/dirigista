"""A stand-in for the inference client.

The classifiers are tested against a fake rather than the live service: unit
tests must be deterministic and free. `tests/test_live.py` covers the real
integration.
"""

import pytest
from typesafe_sdk import SystemOneResponse, Usage


class FakeTypeSafeClient:
    """Records the requests it receives and replays canned answers."""

    def __init__(self, answers: dict):
        self._answers = answers
        self.requests: list[tuple] = []

    async def system_one(self, state, questions, **kwargs):
        self.requests.append((state, questions))
        return SystemOneResponse(
            model="jev-latest", usage=Usage(), answers=self._answers
        )

    @property
    def last_state(self):
        return self.requests[-1][0]

    @property
    def last_question(self):
        return next(iter(self.requests[-1][1].values()))


@pytest.fixture
def fake_client():
    return FakeTypeSafeClient
