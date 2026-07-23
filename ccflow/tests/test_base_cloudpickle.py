import base64
import multiprocessing
import pickle
import textwrap
import traceback
from pathlib import Path
from queue import Empty

import cloudpickle


def _spawn_entrypoint(result_queue, worker, args):
    try:
        result_queue.put(("ok", worker(*args)))
    except Exception:  # noqa: BLE001
        result_queue.put(("error", traceback.format_exc()))


def _run_in_spawned_process(worker, *args, timeout: int = 30):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_spawn_entrypoint, args=(result_queue, worker, args))
    process.start()
    try:
        status, result = result_queue.get(timeout=timeout)
    except Empty:
        process.terminate()
        process.join(timeout=5)
        raise AssertionError(f"{worker.__name__} timed out after {timeout} seconds") from None

    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise AssertionError(f"{worker.__name__} did not exit after returning a result")
    if process.exitcode != 0:
        raise AssertionError(f"{worker.__name__} exited with {process.exitcode}")
    if status == "error":
        raise AssertionError(result)
    return result


def _serialize_payloads(payloads: dict[str, tuple[str, object]]) -> dict[str, dict[str, str]]:
    encoded = {}
    for name, (serializer, value) in payloads.items():
        module = pickle if serializer == "pickle" else cloudpickle
        encoded[name] = {
            "serializer": serializer,
            "payload": base64.b64encode(module.dumps(value, protocol=5)).decode(),
        }
    return encoded


def _load_payload(spec: dict[str, str]):
    module = pickle if spec["serializer"] == "pickle" else cloudpickle
    return module.loads(base64.b64decode(spec["payload"]))


def _load_payloads(encoded_payloads: dict[str, dict[str, str]]) -> dict[str, object]:
    return {name: _load_payload(spec) for name, spec in encoded_payloads.items()}


def _load_payload_without_return(spec: dict[str, str]) -> None:
    _load_payload(spec)


def _create_ccflow_generic_payloads() -> dict[str, dict[str, str]]:
    from collections.abc import Callable
    from typing import ClassVar, Final

    import numpy as np

    from ccflow import GenericContext, GenericResult
    from ccflow.result import DictResult, ListResult
    from ccflow.result.numpy import NumpyResult

    return _serialize_payloads(
        {
            "standard_pickle_result": ("pickle", GenericResult[int](value=5)),
            "generic_result": ("cloudpickle", GenericResult[int](value=6)),
            "generic_context": ("cloudpickle", GenericContext[str](value="abc")),
            "list_result": ("cloudpickle", ListResult[int](value=[1, 2])),
            "dict_result": ("cloudpickle", DictResult[str, float](value={"a": 1.5})),
            "numpy_result": (
                "cloudpickle",
                NumpyResult[np.float64](array=np.array([1.0, 2.0], dtype=np.float64)),
            ),
            "nested_result": (
                "cloudpickle",
                GenericResult[ListResult[int]](value=ListResult[int](value=[1, 2])),
            ),
            "list_alias_result": (
                "cloudpickle",
                GenericResult[list[ListResult[int]]](value=[ListResult[int](value=[1])]),
            ),
            "typing_list_alias_result": (
                "cloudpickle",
                GenericResult[list[ListResult[int]]](value=[ListResult[int](value=[2])]),
            ),
            "callable_alias_result": (
                "cloudpickle",
                GenericResult[Callable[[ListResult[int]], int]](value=lambda result: len(result.value)),
            ),
            "optional_alias_result": (
                "cloudpickle",
                GenericResult[ListResult[int] | None](value=ListResult[int](value=[4])),
            ),
            "classvar_alias_result": (
                "cloudpickle",
                GenericResult[ClassVar[ListResult[int]]](value=ListResult[int](value=[5])),
            ),
            "final_alias_result": (
                "cloudpickle",
                GenericResult[Final[ListResult[int]]](value=ListResult[int](value=[6])),
            ),
            "dict_alias_result": (
                "cloudpickle",
                GenericResult[dict[str, GenericContext[int]]](value={"a": GenericContext[int](value=1)}),
            ),
            "union_result": (
                "cloudpickle",
                GenericResult[GenericContext[int] | None](value=GenericContext[int](value=1)),
            ),
        }
    )


def _assert_ccflow_generic_payloads(encoded_payloads: dict[str, dict[str, str]]) -> None:
    from collections.abc import Callable as AbcCallable
    from typing import ClassVar, Final

    import numpy as np

    from ccflow import GenericContext, GenericResult
    from ccflow.result import DictResult, ListResult
    from ccflow.result.numpy import NumpyResult

    values = _load_payloads(encoded_payloads)

    assert values["standard_pickle_result"] == GenericResult[int](value=5)
    assert type(values["standard_pickle_result"]).__pydantic_generic_metadata__["args"] == (int,)

    assert values["generic_result"] == GenericResult[int](value=6)
    assert values["generic_context"] == GenericContext[str](value="abc")
    assert values["list_result"] == ListResult[int](value=[1, 2])
    assert values["dict_result"] == DictResult[str, float](value={"a": 1.5})

    assert type(values["numpy_result"]) is NumpyResult[np.float64]
    np.testing.assert_array_equal(values["numpy_result"].array, np.array([1.0, 2.0], dtype=np.float64))

    assert values["nested_result"] == GenericResult[ListResult[int]](value=ListResult[int](value=[1, 2]))
    assert type(values["nested_result"]).__pydantic_generic_metadata__["args"] == (ListResult[int],)
    assert type(values["nested_result"].value).__pydantic_generic_metadata__["args"] == (int,)

    assert values["list_alias_result"] == GenericResult[list[ListResult[int]]](value=[ListResult[int](value=[1])])
    assert type(values["list_alias_result"]).__pydantic_generic_metadata__["args"] == (list[ListResult[int]],)
    assert type(values["list_alias_result"].value[0]).__pydantic_generic_metadata__["args"] == (int,)

    assert values["typing_list_alias_result"] == GenericResult[list[ListResult[int]]](value=[ListResult[int](value=[2])])
    assert type(values["typing_list_alias_result"]).__pydantic_generic_metadata__["args"] == (list[ListResult[int]],)

    assert values["callable_alias_result"].value(ListResult[int](value=[1, 2, 3])) == 3
    assert type(values["callable_alias_result"]).__pydantic_generic_metadata__["args"] == (AbcCallable[[ListResult[int]], int],)

    assert values["optional_alias_result"] == GenericResult[ListResult[int] | None](value=ListResult[int](value=[4]))
    assert type(values["optional_alias_result"]).__pydantic_generic_metadata__["args"] == (ListResult[int] | None,)

    assert values["classvar_alias_result"] == GenericResult[ClassVar[ListResult[int]]](value=ListResult[int](value=[5]))
    assert type(values["classvar_alias_result"]).__pydantic_generic_metadata__["args"] == (ClassVar[ListResult[int]],)

    assert values["final_alias_result"] == GenericResult[Final[ListResult[int]]](value=ListResult[int](value=[6]))
    assert type(values["final_alias_result"]).__pydantic_generic_metadata__["args"] == (Final[ListResult[int]],)

    assert values["dict_alias_result"] == GenericResult[dict[str, GenericContext[int]]](value={"a": GenericContext[int](value=1)})
    assert type(values["dict_alias_result"]).__pydantic_generic_metadata__["args"] == (dict[str, GenericContext[int]],)
    assert type(values["dict_alias_result"].value["a"]).__pydantic_generic_metadata__["args"] == (int,)

    assert values["union_result"] == GenericResult[GenericContext[int] | None](value=GenericContext[int](value=1))
    assert type(values["union_result"].value).__pydantic_generic_metadata__["args"] == (int,)


def _assert_cold_ccflow_generic_payload_loads(spec: dict[str, str]) -> None:
    import ccflow.result.generic as generic_module
    from ccflow import GenericResult  # noqa: F401

    had_specialization_before_load = hasattr(generic_module, "GenericResult[int]")
    value = _load_payload(spec)
    assert (
        had_specialization_before_load,
        value.value,
        tuple(arg.__name__ for arg in type(value).__pydantic_generic_metadata__["args"]),
    ) == (False, 6, ("int",))


def _create_user_generic_payloads(module_dir: str) -> dict[str, dict[str, str]]:
    import sys
    from typing import Generic, TypeVar

    from pydantic import PrivateAttr

    from ccflow import BaseModel

    sys.path.insert(0, module_dir)
    from generic_user_model import UserBox

    T = TypeVar("T")

    class LocalBox(BaseModel, Generic[T]):
        value: T
        _bonus: int = PrivateAttr(default=1)

    class LocalPayload(BaseModel):
        value: int
        _bonus: int = PrivateAttr(default=1)

    importable = UserBox[int](value=2)
    importable._bonus = 40
    local_generic = LocalBox[int](value=3)
    local_generic._bonus = 41
    local_payload = LocalPayload(value=4)
    local_payload._bonus = 42

    return _serialize_payloads(
        {
            "importable_generic": ("cloudpickle", importable),
            "local_generic": ("cloudpickle", local_generic),
            "local_payload": ("cloudpickle", local_payload),
        }
    )


def _assert_user_generic_payloads(encoded_payloads: dict[str, dict[str, str]], module_dir: str) -> None:
    import sys

    sys.path.insert(0, module_dir)
    from generic_user_model import UserBox

    values = _load_payloads(encoded_payloads)

    assert type(values["importable_generic"]) is UserBox[int]
    assert values["importable_generic"].value == 2
    assert values["importable_generic"]._bonus == 40

    assert values["local_generic"].value == 3
    assert values["local_generic"]._bonus == 41
    assert type(values["local_generic"]).__name__ == "LocalBox[int]"
    assert type(values["local_generic"]).__pydantic_generic_metadata__["args"] == (int,)

    assert values["local_payload"].value == 4
    assert values["local_payload"]._bonus == 42
    assert type(values["local_payload"]).__name__ == "LocalPayload"


def _load_user_payload(spec: dict[str, str], module_dir: str):
    import sys

    sys.path.insert(0, module_dir)
    return _load_payload(spec)


def _load_user_payload_without_return(spec: dict[str, str], module_dir: str) -> None:
    _load_user_payload(spec, module_dir)


def _assert_payloads_load_in_spawned_process(
    payload_factory,
    assertion_worker,
    *factory_args,
    cold_loader_worker=_load_payload_without_return,
) -> None:
    # The bug only appears when the receiver has not already materialized the
    # same Pydantic generic specialization. Create and load payloads in spawned
    # subprocesses so the pytest process cannot warm the receiver's generic
    # cache by accident.
    encoded_payloads = _run_in_spawned_process(payload_factory, *factory_args)

    # First load each payload in its own fresh process. The original bug was
    # sensitive to whether a worker had already materialized the same generic
    # specialization, so a shared loader process can create false positives.
    for spec in encoded_payloads.values():
        _run_in_spawned_process(cold_loader_worker, spec, *factory_args)

    # Load in a second fresh process, then run all assertions there. Keeping many
    # payloads in one process pair keeps detailed assertions cheap. The
    # per-payload loop above already verifies cold-receiver loading.
    _run_in_spawned_process(assertion_worker, encoded_payloads, *factory_args)


def test_ccflow_generic_specializations_pickle_across_fresh_processes():
    # One matrix test covers the ccflow-provided generic families and the hard
    # type-argument shapes: nested Pydantic generics, builtin aliases containing
    # Pydantic generics, and PEP 604 unions containing Pydantic generics.
    _assert_payloads_load_in_spawned_process(_create_ccflow_generic_payloads, _assert_ccflow_generic_payloads)

    encoded_payloads = _run_in_spawned_process(_create_ccflow_generic_payloads)
    _run_in_spawned_process(_assert_cold_ccflow_generic_payload_loads, encoded_payloads["generic_result"])


def test_user_generic_and_non_generic_ccflow_models_pickle_across_fresh_processes(tmp_path: Path):
    # User subclasses exercise the same BaseModel reducer without depending on
    # ccflow's result/context classes. The non-generic case proves the generic
    # override has not captured ordinary BaseModel pickling.
    module_path = tmp_path / "generic_user_model.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from typing import Generic, TypeVar

            from pydantic import PrivateAttr

            from ccflow import BaseModel

            T = TypeVar("T")

            class UserBox(BaseModel, Generic[T]):
                value: T
                _bonus: int = PrivateAttr(default=1)
            """
        )
    )

    _assert_payloads_load_in_spawned_process(
        _create_user_generic_payloads,
        _assert_user_generic_payloads,
        str(tmp_path),
        cold_loader_worker=_load_user_payload_without_return,
    )
