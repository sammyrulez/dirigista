"""Classification logic.

This module knows nothing about MCP. It builds a question for the inference
backend, sends it through an injected client, and turns the typed answer into
one of the models in `models.py`. The client is a parameter rather than a global
so the conversion logic can be tested without the network.

The backend answers with probabilities, never prose. Everything these functions
return is therefore either a value the model produced or an explicit, documented
derivation from one — never an explanation composed after the fact.
"""

from typing import Protocol

from typesafe_sdk import Choice, Noul, Score

from .models import CategorizeResult, ScoreResult, VerifyResult


class SystemOneClient(Protocol):
    """The slice of the inference client these functions depend on."""

    async def system_one(self, state, questions, **kwargs): ...


async def verify(
    statement: str,
    context: str | None = None,
    *,
    client: SystemOneClient,
    false_below: float = 0.2,
    true_above: float = 0.8,
) -> VerifyResult:
    """Decide whether `statement` is true, false, or undecidable.

    The model returns one probability; the verdict is that probability read
    against `false_below` and `true_above`. The defaults are a convention, not a
    truth about the domain — a caller who pays more for a false positive than a
    false negative should move them.
    """
    state: dict[str, str] = {"statement": statement}
    if context is not None:
        state["context"] = context

    response = await client.system_one(
        state=state,
        questions={
            "verify": Noul(
                instructions=(
                    "Is the claim in `statement` true? Judge the claim itself, "
                    "not whether it is well phrased or politely put."
                    + (" Use `context` as the situation it is made in." if context else "")
                ),
                criteria={
                    "true": "The claim holds as stated.",
                    "false": "The claim does not hold as stated.",
                },
            )
        },
    )

    probability = response.nouls["verify"].noul
    if probability >= true_above:
        verdict = "true"
    elif probability <= false_below:
        verdict = "false"
    else:
        verdict = "uncertain"

    return VerifyResult(verdict=verdict, probability=probability)


async def categorize(
    statement: str,
    categories: list[str],
    *,
    client: SystemOneClient,
) -> CategorizeResult:
    """Pick the category from `categories` that best fits `statement`."""
    if len(categories) < 2:
        raise ValueError("categorize needs at least two categories to choose between")

    response = await client.system_one(
        state={"statement": statement},
        questions={
            "categorize": Choice(
                instructions=(
                    "Which category best describes `statement`? Choose the single "
                    "closest fit."
                ),
                criteria={category: None for category in categories},
            )
        },
    )

    answer = response.choices["categorize"]
    return CategorizeResult(
        category=answer.choice,
        confidence=answer.confidence,
        probabilities=answer.probabilities,
    )


async def score(
    statement: str,
    criteria: list[str],
    *,
    client: SystemOneClient,
) -> ScoreResult:
    """Rate `statement` against `criteria`, an ordinal scale ordered low to high."""
    if len(criteria) < 2:
        raise ValueError("score needs at least two criteria to form a scale")

    response = await client.system_one(
        state={"statement": statement},
        questions={
            "score": Score(
                instructions=(
                    "Where on this scale does `statement` sit? The levels run "
                    "from lowest to highest."
                ),
                criteria=list(criteria),
            )
        },
    )

    answer = response.scores["score"]
    # The model's raw score is a probability-weighted mean and can land between
    # levels; the caller asked for a level, so report the most probable one
    # rather than rounding the mean, which would answer a different question.
    index = max(answer.probabilities, key=lambda level: answer.probabilities[level])
    return ScoreResult(
        criterion=criteria[index],
        level=index + 1,
        confidence=answer.confidence,
    )
