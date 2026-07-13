import sys
from unittest.mock import patch

import pytest

from ccflow import Flow, FlowOptionsOverride, FromContext, ModelRegistry
from ccflow.evaluators import GraphEvaluator, MemoryCacheEvaluator, MultiEvaluator
from ccflow.examples.calculator import load_config
from ccflow.examples.calculator.__main__ import main
from ccflow.examples.calculator.explain import explain
from ccflow.examples.calculator.functions import Numbers, add, lower_gap, power, rounded, scale, tail_ratio, upper_gap


class TestCalculatorFunctions:
    def test_add(self):
        model = add(offset=1.0)
        assert model(Numbers(values=[1, 2, 3])).value == 7.0

    def test_scale(self):
        model = scale(factor=10.0)
        assert model(Numbers(values=[1, 2, 3])).value == 60.0

    def test_power(self):
        model = power(exponent=3.0)
        assert model(Numbers(values=[1, 2, 3])).value == 36.0

    def test_context_from_dict(self):
        # cfg_run passes the context as a plain dict; it validates to Numbers.
        model = add()
        assert model({"values": [1, 2, 3]}).value == 6.0

    def test_composition_derives_deps(self):
        # @Flow.model generates __deps__ from the wiring; the graph evaluator runs it.
        model = rounded(value=power(exponent=2.0))
        assert hasattr(model, "__deps__")
        assert model.flow.inspect().bound_inputs["value"] is not None
        evaluator = MultiEvaluator(evaluators=[GraphEvaluator(), MemoryCacheEvaluator()])
        with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
            assert model({"values": [1.5, 2.5]}).value == 8.5

    def test_diamond_dedup(self):
        # tail_ratio needs upper_gap and lower_gap, which both need the mean:
        # a diamond whose shared node the graph evaluator computes once.
        calls = {"mean": 0}

        @Flow.model(context_type=Numbers)
        def counting_mean(values: FromContext[list[float]]) -> float:
            calls["mean"] += 1
            return sum(values) / len(values)

        center = counting_mean()
        model = tail_ratio(upper=upper_gap(center=center), lower=lower_gap(center=center))

        model({"values": [1, 2, 3, 10]})
        assert calls["mean"] == 2  # once per branch under plain evaluation

        calls["mean"] = 0
        evaluator = MultiEvaluator(evaluators=[GraphEvaluator(), MemoryCacheEvaluator()])
        with FlowOptionsOverride(options={"cacheable": True, "evaluator": evaluator}):
            assert model({"values": [1, 2, 3, 10]}).value == 2.0
        assert calls["mean"] == 1  # shared node evaluated once


class TestCalculatorConfig:
    def _run(self, overrides, values):
        try:
            registry = load_config(overrides=overrides)
            return registry["/function"]({"values": values}).value
        finally:
            ModelRegistry.root().clear()

    def test_default_function(self):
        assert self._run([], [1, 2, 3]) == 6.0

    def test_swap_function(self):
        assert self._run(["function=scale", "function.factor=10"], [1, 2, 3]) == 60.0

    def test_configure_function(self):
        assert self._run(["function=power", "function.exponent=3"], [1, 2, 3]) == 36.0

    def test_composed_function(self):
        # `rounded` wraps `power` (exponent 2) as an upstream model input.
        assert self._run(["function=rounded", "function.digits=1"], [1.5, 2.5]) == 8.5

    def test_diamond_function(self):
        # tail_ratio composes a shared-mean diamond entirely in YAML.
        assert self._run(["function=tail_ratio"], [1, 2, 3, 10]) == 2.0


class TestCalculatorOutput:
    def _run_output(self, overrides, values):
        try:
            registry = load_config(overrides=overrides)
            return registry["/output"]({"values": values}).value
        finally:
            ModelRegistry.root().clear()

    def test_output_print(self, capsys):
        result = self._run_output(["output=print", "function=power", "function.exponent=3"], [1, 2, 3])
        assert result == 36.0
        assert "36.0" in capsys.readouterr().out

    def test_output_log(self):
        result = self._run_output(["output=log"], [1, 2, 3])
        assert result == 6.0

    def test_output_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            registry = load_config(overrides=["output=write", "function=scale", "function.factor=10"])
            registry["/output"]({"values": [1, 2, 3]})
        finally:
            ModelRegistry.root().clear()
        assert (tmp_path / "result.txt").read_text() == "60.0"


class TestCalculatorCli:
    @pytest.mark.skipif(sys.version_info >= (3, 14), reason="Hydra shell completion help string incompatible with Python 3.14 argparse")
    def test_cli(self):
        with patch("ccflow.examples.calculator.__main__.cfg_run") as mock_cfg_run:
            with patch("sys.argv", ["calculator", "function=power", "+context.values=[1,2,3]"]):
                main()
                mock_cfg_run.assert_called_once()

    def test_explain(self):
        with patch("ccflow.examples.calculator.explain.cfg_explain_cli") as mock_cfg_explain:
            explain()
            mock_cfg_explain.assert_called_once()
