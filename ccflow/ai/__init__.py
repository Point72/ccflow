"""An adversarial review graph built on ccflow: write a poem, argue both sides, adjudicate.

One agent produces an artifact, two argue opposing positions about it, and a fourth decides which
argued better. Every hand-off is a typed pydantic contract, mechanical constraints are enforced in
code rather than by asking a model, and the arbiter is insulated from the presentation effects that
were measured to move its decisions.

Requires the optional ``ai`` extra::

    pip install ccflow[ai]

Run it from the command line, which is where the config groups are steered::

    python -m ccflow.ai checks=off              # offline, needs no credentials
    python -m ccflow.ai model=openai            # against a real model
    python -m ccflow.ai task=counterbalanced    # judge both presentation orders

Or load it into the registry, as the other bundled examples do::

    from ccflow import ModelRegistry
    from ccflow.ai import load_config

    load_config(overrides=["checks=off"])
    result = ModelRegistry.root()["/task"]()
    print(result.verdict.winner, result.verdict.decisive)
"""

from pathlib import Path

from ccflow import RootModelRegistry, load_config as _load_config_base

from .agents import AgentProfile, load_agent_profile, profile_search_roots
from .checks import ClaimsWithinLimit, EvidenceAppearsInArtifact, LinesWithinLimit, OutputCheck, PositionAsAssigned, VerdictWellFormed, run_checks
from .contracts import (
    ArbitrationBrief,
    ArbitrationRubric,
    Claim,
    Critique,
    CritiqueBrief,
    CritiqueRubric,
    Lessons,
    Poem,
    PoemSpec,
    PositionScore,
    RebuttalBrief,
    Verdict,
)
from .graph import AdvocateModel, ArbitrateModel, CounterbalancedArbitrateModel, DebateModel, ProduceModel, RebuttalModel
from .results import CritiqueProvider, CritiqueResult, PoemProvider, PoemResult, PublishingModel, VerdictProvider, VerdictResult
from .session import AgentSession, AgentSessionContext, AgentSessionResult
from .sources import CritiqueFileSource, PoemFileSource

__all__ = (
    "AdvocateModel",
    "AgentProfile",
    "AgentSession",
    "AgentSessionContext",
    "AgentSessionResult",
    "ArbitrateModel",
    "ArbitrationBrief",
    "ArbitrationRubric",
    "Claim",
    "ClaimsWithinLimit",
    "CounterbalancedArbitrateModel",
    "Critique",
    "CritiqueBrief",
    "CritiqueFileSource",
    "CritiqueProvider",
    "CritiqueResult",
    "CritiqueRubric",
    "DebateModel",
    "EvidenceAppearsInArtifact",
    "Lessons",
    "LinesWithinLimit",
    "OutputCheck",
    "Poem",
    "PoemFileSource",
    "PoemProvider",
    "PoemResult",
    "PoemSpec",
    "PositionAsAssigned",
    "PositionScore",
    "ProduceModel",
    "PublishingModel",
    "RebuttalBrief",
    "RebuttalModel",
    "Verdict",
    "VerdictProvider",
    "VerdictResult",
    "VerdictWellFormed",
    "load_agent_profile",
    "load_config",
    "profile_search_roots",
    "run_checks",
)


def load_config(
    config_dir: str = "",
    config_name: str = "",
    overrides: list[str] | None = None,
    *,
    overwrite: bool = True,
    basepath: str = "",
) -> RootModelRegistry:
    """Load the poem review example into the root ``ModelRegistry``.

    Args:
        config_dir: Optional extra hydra config directory overlaid on the bundled ``config``.
            Empty string (the default) means "use only the bundled config".
        config_name: Optional config name within ``config_dir`` to load.
        overrides: Hydra override strings selecting config groups, e.g. ``["task=counterbalanced"]``.
        overwrite: When True (the default), entries already in the registry are replaced.
        basepath: Base path for resolving a relative ``config_dir``.
    """
    return _load_config_base(
        root_config_dir=str(Path(__file__).resolve().parent / "config"),
        root_config_name="base",
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
        overwrite=overwrite,
        basepath=basepath,
    )
