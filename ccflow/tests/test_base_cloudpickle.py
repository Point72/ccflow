import json
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)


def _create_payloads_in_fresh_process(creator_body: str) -> dict[str, dict[str, str]]:
    # The bug only appears when the receiver has not already materialized the
    # same Pydantic generic specialization. Create payloads in a subprocess so
    # the pytest process cannot accidentally warm the receiver's generic cache.
    creator = _run_python(
        "import base64\n"
        "import cloudpickle\n"
        "import json\n"
        "import pickle\n"
        f"{textwrap.dedent(creator_body)}\n"
        "encoded = {}\n"
        "for name, (serializer, value) in payloads.items():\n"
        "    module = pickle if serializer == 'pickle' else cloudpickle\n"
        "    encoded[name] = {\n"
        "        'serializer': serializer,\n"
        "        'payload': base64.b64encode(module.dumps(value, protocol=5)).decode(),\n"
        "    }\n"
        "print(json.dumps(encoded, sort_keys=True))\n"
    )
    assert creator.returncode == 0, creator.stderr
    return json.loads(creator.stdout)


def _assert_payloads_load_in_fresh_process(creator_body: str, assertions: str, *, extra_loader_setup: str = "") -> None:
    encoded_payloads = _create_payloads_in_fresh_process(creator_body)
    # First load each payload in its own fresh process. The original bug was
    # sensitive to whether a worker had already materialized the same generic
    # specialization, so a shared loader process can accidentally warm the
    # receiver and create false positives.
    for name, spec in encoded_payloads.items():
        cold_loader = _run_python(
            "import base64\n"
            "import cloudpickle\n"
            "import json\n"
            "import pickle\n"
            f"{textwrap.dedent(extra_loader_setup)}\n"
            f"spec = json.loads({json.dumps(spec)!r})\n"
            "module = pickle if spec['serializer'] == 'pickle' else cloudpickle\n"
            "module.loads(base64.b64decode(spec['payload']))\n"
        )
        assert cold_loader.returncode == 0, f"{name}: {cold_loader.stderr}"

    # Load in a second fresh process, then run all assertions there. Keeping many
    # payloads in one process pair keeps the detailed assertions cheap. The
    # per-payload loop above already verifies cold-receiver loading.
    loader = _run_python(
        "import base64\n"
        "import cloudpickle\n"
        "import json\n"
        "import pickle\n"
        f"{textwrap.dedent(extra_loader_setup)}\n"
        f"encoded = json.loads({json.dumps(encoded_payloads)!r})\n"
        "values = {}\n"
        "for name, spec in encoded.items():\n"
        "    module = pickle if spec['serializer'] == 'pickle' else cloudpickle\n"
        "    values[name] = module.loads(base64.b64decode(spec['payload']))\n"
        f"{textwrap.dedent(assertions)}\n"
    )
    assert loader.returncode == 0, loader.stderr


def test_ccflow_generic_specializations_pickle_across_fresh_processes():
    # One matrix test covers the ccflow-provided generic families and the hard
    # type-argument shapes: nested Pydantic generics, builtin aliases containing
    # Pydantic generics, and PEP 604 unions containing Pydantic generics.
    _assert_payloads_load_in_fresh_process(
        """
        import numpy as np
        from typing import Callable, ClassVar, Final, List, Optional

        from ccflow import GenericContext, GenericResult
        from ccflow.result import DictResult, ListResult
        from ccflow.result.numpy import NumpyResult

        payloads = {
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
                GenericResult[List[ListResult[int]]](value=[ListResult[int](value=[2])]),
            ),
            "callable_alias_result": (
                "cloudpickle",
                GenericResult[Callable[[ListResult[int]], int]](value=lambda result: len(result.value)),
            ),
            "optional_alias_result": (
                "cloudpickle",
                GenericResult[Optional[ListResult[int]]](value=ListResult[int](value=[4])),
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
        """,
        """
        import numpy as np
        from typing import Callable, ClassVar, Final, List, Optional

        from ccflow import GenericContext, GenericResult
        from ccflow.result import DictResult, ListResult
        from ccflow.result.numpy import NumpyResult

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

        assert values["typing_list_alias_result"] == GenericResult[List[ListResult[int]]](
            value=[ListResult[int](value=[2])]
        )
        assert type(values["typing_list_alias_result"]).__pydantic_generic_metadata__["args"] == (List[ListResult[int]],)

        assert values["callable_alias_result"].value(ListResult[int](value=[1, 2, 3])) == 3
        assert type(values["callable_alias_result"]).__pydantic_generic_metadata__["args"] == (
            Callable[[ListResult[int]], int],
        )

        assert values["optional_alias_result"] == GenericResult[Optional[ListResult[int]]](
            value=ListResult[int](value=[4])
        )
        assert type(values["optional_alias_result"]).__pydantic_generic_metadata__["args"] == (
            Optional[ListResult[int]],
        )

        assert values["classvar_alias_result"] == GenericResult[ClassVar[ListResult[int]]](
            value=ListResult[int](value=[5])
        )
        assert type(values["classvar_alias_result"]).__pydantic_generic_metadata__["args"] == (
            ClassVar[ListResult[int]],
        )

        assert values["final_alias_result"] == GenericResult[Final[ListResult[int]]](
            value=ListResult[int](value=[6])
        )
        assert type(values["final_alias_result"]).__pydantic_generic_metadata__["args"] == (
            Final[ListResult[int]],
        )

        assert values["dict_alias_result"] == GenericResult[dict[str, GenericContext[int]]](
            value={"a": GenericContext[int](value=1)}
        )
        assert type(values["dict_alias_result"]).__pydantic_generic_metadata__["args"] == (
            dict[str, GenericContext[int]],
        )
        assert type(values["dict_alias_result"].value["a"]).__pydantic_generic_metadata__["args"] == (int,)

        assert values["union_result"] == GenericResult[GenericContext[int] | None](value=GenericContext[int](value=1))
        assert type(values["union_result"].value).__pydantic_generic_metadata__["args"] == (int,)
        """,
    )


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

    _assert_payloads_load_in_fresh_process(
        f"""
        import sys
        from typing import Generic, TypeVar

        from pydantic import PrivateAttr

        from ccflow import BaseModel

        sys.path.insert(0, {str(tmp_path)!r})
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

        payloads = {{
            "importable_generic": ("cloudpickle", importable),
            "local_generic": ("cloudpickle", local_generic),
            "local_payload": ("cloudpickle", local_payload),
        }}
        """,
        """
        from generic_user_model import UserBox

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
        """,
        extra_loader_setup=f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        """,
    )


def test_ccflow_generic_result_cloudpickle_in_ray_worker_without_receiver_materialization():
    import ray

    # Ray is the production-shaped failure mode: workers are fresh processes and
    # may deserialize before ``GenericResult[int]`` has ever been created there.
    payload = _create_payloads_in_fresh_process(
        """
        from ccflow import GenericResult

        payloads = {"generic_result": ("cloudpickle", GenericResult[int](value=5))}
        """
    )["generic_result"]["payload"]

    @ray.remote
    def load_payload(encoded_payload: str):
        import base64

        import cloudpickle

        import ccflow.result.generic as generic_module
        from ccflow import GenericResult  # noqa: F401

        # Importing the generic origin must not be enough to make the test pass.
        # The worker should not know about ``GenericResult[int]`` until the
        # reducer intentionally rebuilds it during cloudpickle.loads.
        had_specialization_before_load = hasattr(generic_module, "GenericResult[int]")
        value = cloudpickle.loads(base64.b64decode(encoded_payload))
        return (
            had_specialization_before_load,
            value.value,
            tuple(arg.__name__ for arg in type(value).__pydantic_generic_metadata__["args"]),
        )

    with ray.init(num_cpus=1):
        assert ray.get(load_payload.remote(payload), timeout=30) == (False, 5, ("int",))
