"""Loading agent personas from markdown.

Personas are markdown files with optional YAML frontmatter, kept beside the code rather than inlined
as Python strings so they can be read, diffed and edited as prose. The body below the frontmatter is
what the model is given.

``CCFLOW_AI_AGENTS_PATH`` (os.pathsep-separated) prepends extra directories to the search, which is
how a caller supplies its own personas without forking this package.
"""

from functools import cache
from os import environ, pathsep
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

__all__ = ("AgentProfile", "load_agent_profile", "profile_search_roots")

_FRONTMATTER_DELIMITER = "---"
_ENV_VAR = "CCFLOW_AI_AGENTS_PATH"


class AgentProfile(BaseModel):
    """A persona: its description and the instructions handed to the model."""

    name: str = Field(description="Profile name, the markdown file stem.")
    description: str = Field(default="", description="One line on what this persona is for.")
    instructions: str = Field(description="The markdown body, given to the model as its persona.")


def profile_search_roots() -> list[Path]:
    """Directories searched for persona files, highest precedence first."""
    roots: list[Path] = []
    if override := environ.get(_ENV_VAR):
        roots.extend(Path(p) for p in override.split(pathsep) if p)
    roots.append(Path(__file__).resolve().parent / "personas")
    return roots


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.lstrip().startswith(_FRONTMATTER_DELIMITER):
        return {}, text
    _, _, rest = text.lstrip().partition(_FRONTMATTER_DELIMITER)
    front, delimiter, body = rest.partition(f"\n{_FRONTMATTER_DELIMITER}")
    if not delimiter:
        return {}, text
    return yaml.safe_load(front) or {}, body


@cache
def load_agent_profile(name: str) -> AgentProfile:
    """Read one persona by file stem, searching `profile_search_roots` in order."""
    roots = profile_search_roots()
    for root in roots:
        path = root / f"{name}.md"
        if path.is_file():
            metadata, body = _split_frontmatter(path.read_text(encoding="utf-8"))
            return AgentProfile(name=name, description=metadata.get("description", ""), instructions=body.strip())
    searched = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(f"No persona {name!r}.md found. Searched: {searched}. Set {_ENV_VAR} to add directories.")
