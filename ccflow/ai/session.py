"""The agent session: one ccflow `CallableModel`, one typed language-model call.

This is the only place in the subpackage that talks to a model. Every node composes one of these.

The output contract is named by dotted path rather than held as a class, because the graph is
configured from YAML and Hydra cannot express a Python type. Resolution happens once, at call time,
and fails loudly if the path is not a pydantic model — a mistyped contract should not surface as a
confusing validation error three nodes later.

Structured-output retries are delegated to pydantic-ai: on a schema violation it re-prompts with the
validation error attached, which repairs far more reliably than a blank retry.

``pydantic-ai`` is imported inside the methods that need it, so the contracts, checks and
configuration here stay importable without the optional `ai` extra installed.

This module deliberately omits ``from __future__ import annotations``: ccflow resolves the
``__call__`` context type by ``issubclass``, and postponed evaluation leaves it a string.
"""

import importlib
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ccflow import CallableModel, ContextBase, Flow, ResultBase

from .agents import AgentProfile, load_agent_profile
from .checks import OutputCheck, run_checks
from .contracts import Lessons

__all__ = ("AgentSession", "AgentSessionContext", "AgentSessionResult")


class AgentSessionContext(ContextBase):
    """What the session is asked to act on: a rendered prompt, and any mechanical constraints.

    Checks travel on the context rather than on the model because they depend on the run — the line
    budget comes from the specification, the evidence check needs the poem text — while the session
    itself is configuration.
    """

    prompt: str = Field(description="The rendered contract instance the model is asked to act on.")
    checks: list[OutputCheck] = Field(default_factory=list, description="Mechanical constraints enforced before the output is accepted.")


class AgentSessionResult(ResultBase):
    """The validated contract instance, plus what it cost to get it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = Field(description="Instance of the configured output contract.")
    model: str = Field(description="Model identifier that produced it.")
    role: str = Field(description="Role this session played.")
    profile_name: str = Field(description="Persona that played the role.")
    input_tokens: int | None = None
    output_tokens: int | None = None
    requests: int | None = Field(default=None, description="Model requests made, which exceeds one when output had to be repaired.")


def _resolve_contract(path: str) -> type[BaseModel]:
    """Import a dotted path and check it really is a pydantic model."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Output contract {path!r} must be a full dotted path, e.g. 'ccflow.ai.contracts.Poem'.")
    obj = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise TypeError(f"Output contract {path!r} resolved to {obj!r}, which is not a pydantic BaseModel.")
    return obj


class AgentSession(CallableModel):
    """One agent turn: a persona and a command, held to a typed output contract."""

    model: str = Field(
        default="test",
        description="Model identifier understood by pydantic-ai, e.g. 'openai:gpt-4o-mini'. The default, 'test', is its TestModel: it fills the "
        "output contract with generated data and needs no credentials, so the graph runs offline.",
    )
    role: str = Field(description="Which node this session plays: producer, pro, con, arbiter. Selects role-scoped lessons.")
    profile_name: str = Field(
        description="Persona file stem supplying the instructions. Kept separate from `role` so swapping personas keeps lessons."
    )
    command: str = Field(description="What this node is asked to do on this run.")
    output_contract: str = Field(description="Dotted path to the pydantic model the output must satisfy.")
    lessons: Lessons = Field(default_factory=Lessons, description="Standing corrections from earlier runs. Only those matching `role` are injected.")
    max_repair_attempts: int = Field(default=2, ge=0, description="Re-prompts allowed when output fails its contract schema or a check.")
    model_settings: dict[str, Any] = Field(default_factory=dict, description="Extra settings forwarded to the model.")

    @cached_property
    def profile(self) -> AgentProfile:
        return load_agent_profile(self.profile_name)

    @cached_property
    def contract(self) -> type[BaseModel]:
        return _resolve_contract(self.output_contract)

    @cached_property
    def instructions(self) -> str:
        """Persona, this run's command, and any lessons scoped to this role."""
        parts = [self.profile.instructions, f"# Your task\n\n{self.command}"]
        if applicable := self.lessons.for_role(self.role):
            body = "\n".join(f"- {lesson}" for lesson in applicable)
            parts.append(f"# Lessons from earlier runs\n\nStanding corrections. Apply every one.\n\n{body}")
        return "\n\n".join(part.strip() for part in parts if part.strip())

    def _build_agent(self, checks: list[OutputCheck] | None = None):
        from pydantic_ai import Agent, ModelRetry

        agent = Agent(self.model, instructions=self.instructions, output_type=self.contract, retries=self.max_repair_attempts)

        if checks:

            @agent.output_validator
            def _enforce(output):
                if problems := run_checks(checks, output):
                    raise ModelRetry(
                        "Your output violates these constraints: " + "; ".join(problems) + ". Correct all of them and return the output again."
                    )
                return output

        return agent

    @Flow.call
    def __call__(self, context: AgentSessionContext) -> AgentSessionResult:
        run = self._build_agent(context.checks).run_sync(context.prompt, model_settings=self.model_settings or None)
        usage = run.usage
        return AgentSessionResult(
            output=run.output,
            model=self.model,
            role=self.role,
            profile_name=self.profile_name,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            requests=getattr(usage, "requests", None),
        )
