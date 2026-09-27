"""Behaviour of the classifiers: how a model answer becomes a tool result."""

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from dirigista import classifiers


# --- verify -----------------------------------------------------------------


async def _verify(client_factory, noul, **kwargs):
    client = client_factory({"verify": NoulAnswer(noul=noul)})
    result = await classifiers.verify("the sky is blue", client=client, **kwargs)
    return result, client


@pytest.mark.parametrize(
    ("noul", "verdict"),
    [(0.97, "true"), (0.03, "false"), (0.5, "uncertain")],
)
async def test_verify_turns_the_probability_into_a_verdict(
    fake_client, noul, verdict
):
    result, _ = await _verify(fake_client, noul)

    assert result.verdict == verdict


async def test_verify_keeps_the_raw_probability_alongside_the_verdict(fake_client):
    result, _ = await _verify(fake_client, 0.97)

    assert result.probability == 0.97


async def test_verify_thresholds_are_callable_choices_not_server_policy(fake_client):
    result, _ = await _verify(fake_client, 0.6, false_below=0.7, true_above=0.95)

    assert result.verdict == "false"


async def test_verify_sends_the_statement_as_state(fake_client):
    _, client = await _verify(fake_client, 0.9)

    assert client.last_state["statement"] == "the sky is blue"


async def test_verify_sends_the_context_only_when_the_caller_supplies_one(fake_client):
    _, without = await _verify(fake_client, 0.9)
    _, with_context = await _verify(fake_client, 0.9, context="it is midnight")

    assert "context" not in without.last_state
    assert with_context.last_state["context"] == "it is midnight"


# --- categorize -------------------------------------------------------------


async def _categorize(client_factory, choice, categories, confidence=0.8):
    probabilities = {c: 0.0 for c in categories} | {choice: confidence}
    client = client_factory(
        {
            "categorize": ChoiceAnswer(
                choice=choice, confidence=confidence, probabilities=probabilities
            )
        }
    )
    result = await classifiers.categorize("hello", categories, client=client)
    return result, client


async def test_categorize_returns_the_chosen_category_and_its_confidence(fake_client):
    result, _ = await _categorize(fake_client, "greeting", ["greeting", "complaint"])

    assert result.category == "greeting"
    assert result.confidence == 0.8


async def test_categorize_exposes_the_full_distribution(fake_client):
    result, _ = await _categorize(fake_client, "greeting", ["greeting", "complaint"])

    assert set(result.probabilities) == {"greeting", "complaint"}


async def test_categorize_offers_every_caller_category_to_the_model(fake_client):
    _, client = await _categorize(fake_client, "greeting", ["greeting", "complaint"])

    assert set(client.last_question.criteria) >= {"greeting", "complaint"}


async def test_categorize_rejects_an_empty_category_list(fake_client):
    with pytest.raises(ValueError, match="at least two"):
        await classifiers.categorize("hello", [], client=fake_client({}))


# --- score ------------------------------------------------------------------


async def _score(client_factory, criteria, probabilities, score=1.0):
    legend = {i: c for i, c in enumerate(criteria)}
    client = client_factory(
        {
            "score": ScoreAnswer(
                score=score,
                confidence=0.7,
                legend=legend,
                probabilities=probabilities,
            )
        }
    )
    result = await classifiers.score("I am upset", criteria, client=client)
    return result, client


SCALE = ["Calm, just stating facts", "Frustrated but civil", "Very angry"]


async def test_score_returns_the_most_probable_level_not_the_weighted_mean(
    fake_client,
):
    result, _ = await _score(
        fake_client, SCALE, probabilities={0: 0.1, 1: 0.2, 2: 0.7}, score=1.6
    )

    assert result.criterion == "Very angry"


async def test_score_reports_the_level_as_a_one_based_position(fake_client):
    result, _ = await _score(
        fake_client, SCALE, probabilities={0: 0.1, 1: 0.2, 2: 0.7}, score=1.6
    )

    assert result.level == 3


async def test_score_sends_the_criteria_in_the_order_the_caller_gave_them(fake_client):
    _, client = await _score(fake_client, SCALE, probabilities={0: 1.0, 1: 0.0, 2: 0.0})

    assert list(client.last_question.criteria) == SCALE


async def test_score_rejects_a_scale_with_a_single_level(fake_client):
    with pytest.raises(ValueError, match="at least two"):
        await classifiers.score("hello", ["only one"], client=fake_client({}))
