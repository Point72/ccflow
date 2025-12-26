"""Shared helpers for constructing local-scope contexts/models in tests."""

from typing import ClassVar, Dict, Tuple, Type

from ccflow import CallableModel, ContextBase, Flow, GenericResult, GraphDepList, NullContext


def build_meta_sensor_planner():
    """Return a (SensorQuery, MetaSensorPlanner, captured) tuple for meta-callable tests."""

    captured: Dict[str, Type] = {}

    class SensorQuery(ContextBase):
        sensor_type: str
        site: str
        window: int

    class MetaSensorPlanner(CallableModel):
        warm_start: int = 2

        @Flow.call
        def __call__(self, context: SensorQuery) -> GenericResult:
            # Define request-scoped specialist wiring with a bespoke context/model pair.
            class SpecialistContext(ContextBase):
                sensor_type: str
                window: int
                pipeline: str

            class SpecialistCallable(CallableModel):
                pipeline: str

                @Flow.call
                def __call__(self, context: SpecialistContext) -> GenericResult:
                    payload = f"{self.pipeline}:{context.sensor_type}:{context.window}"
                    return GenericResult(value=payload)

            captured["context_cls"] = SpecialistContext
            captured["callable_cls"] = SpecialistCallable

            window = context.window + self.warm_start
            local_context = SpecialistContext(
                sensor_type=context.sensor_type,
                window=window,
                pipeline=f"{context.site}-calibration",
            )
            specialist = SpecialistCallable(pipeline=f"planner:{context.site}")
            return specialist(local_context)

    return SensorQuery, MetaSensorPlanner, captured


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
