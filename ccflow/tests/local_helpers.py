"""Shared helpers for constructing local-scope contexts/models in tests."""

from typing import ClassVar, Tuple, Type

from ccflow import CallableModel, ContextBase, Flow, GenericResult, GraphDepList, NullContext


def build_local_callable(name: str = "LocalCallable") -> Type[CallableModel]:
    class _LocalCallable(CallableModel):
        @Flow.call
        def __call__(self, context: NullContext) -> GenericResult:
            return GenericResult(value="local")

    _LocalCallable.__name__ = name
    return _LocalCallable


def build_local_context(name: str = "LocalContext") -> Type[ContextBase]:
    class _LocalContext(ContextBase):
        value: int

    _LocalContext.__name__ = name
    return _LocalContext


def build_nested_graph_chain() -> Tuple[Type[CallableModel], Type[CallableModel]]:
    class LocalLeaf(CallableModel):
        call_count: ClassVar[int] = 0

        @Flow.call
        def __call__(self, context: NullContext) -> GenericResult:
            type(self).call_count += 1
            return GenericResult(value="leaf")

    class LocalParent(CallableModel):
        child: LocalLeaf

        @Flow.call
        def __call__(self, context: NullContext) -> GenericResult:
            return self.child(context)

        @Flow.deps
        def __deps__(self, context: NullContext) -> GraphDepList:
            return [(self.child, [context])]

    return LocalParent, LocalLeaf
