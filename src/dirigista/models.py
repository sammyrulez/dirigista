"""Output contracts for the classification tools.

These models are the server's public surface: MCP clients see them as JSON
Schema, so changes here are breaking changes for callers.

Note what is absent: none of them carries a `reasoning` field. The backend
returns typed judgments and probabilities, not prose, so a rationale string
would have to be invented — and an invented rationale is worse than none.
"""

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["true", "false", "uncertain"]


class VerifyResult(BaseModel):
    """Outcome of a truth check."""

    # No description: the enum lists the three values and the tool docstring
    # explains when 'uncertain' is returned. Repeating it costs context on
    # every request.
    verdict: Verdict
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="That the claim is true. Near 0.5 means genuinely undecided.",
    )


class CategorizeResult(BaseModel):
    """The best-fitting category."""

    category: str = Field(description="One you supplied.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Low when several categories scored alike."
    )
    probabilities: dict[str, float] = Field(description="Sums to 1.")


class ScoreResult(BaseModel):
    """The best-fitting level."""

    criterion: str = Field(description="One you supplied.")
    level: int = Field(ge=1, description="Its 1-based position on your scale.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Low when adjacent levels scored alike."
    )
