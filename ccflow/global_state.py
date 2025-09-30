from typing import Dict, Optional

from pydantic import Field

from .base import BaseModel, ModelRegistry
from .callable import FlowOptionsOverride

__all__ = ("GlobalState",)


class GlobalState(BaseModel):
    """Representation of the global state of the ccflow library.

    Useful when running ccflow functions in remote processes, e.g. with Ray.
    """

    root_registry: ModelRegistry = Field(default_factory=lambda: ModelRegistry.root().clone(name="_"))
    open_overrides: Dict[int, FlowOptionsOverride] = Field(default_factory=lambda: FlowOptionsOverride._OPEN_OVERRIDES.copy())
    _old_state: Optional["GlobalState"] = None

    @classmethod
    def set(cls, state: "GlobalState"):
        root = ModelRegistry.root()
        root.clear()
        for name, model in state.root_registry.models.items():
            root.add(name, model)

        FlowOptionsOverride._OPEN_OVERRIDES = state.open_overrides

    def __enter__(self):
        self._old_state = GlobalState()
        GlobalState.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._old_state is not None:
            GlobalState.set(self._old_state)
        self._old_state = None
