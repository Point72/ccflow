import hydra

from ccflow.utils.hydra import cfg_run

__all__ = ("main",)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    cfg_run(cfg)


# Run the default calculator (add) on some input:
# python -m ccflow.examples.calculator +context.values=[1,2,3]
#
# Swap the calculation (function is in the defaults list, so no `+`):
# python -m ccflow.examples.calculator function=scale +context.values=[1,2,3]
#
# Configure the selected function's field:
# python -m ccflow.examples.calculator function=scale function.factor=10 +context.values=[1,2,3]
# python -m ccflow.examples.calculator function=power function.exponent=3 +context.values=[1,2,3]
#
# A composed calculation (round the result of power):
# python -m ccflow.examples.calculator function=rounded function.digits=1 +context.values=[1.5,2.5]
#
# A diamond that reuses a shared `mean` node, deduped by the graph + cache evaluator:
# python -m ccflow.examples.calculator function=tail_ratio +context.values=[1,2,3,10]
#
# Inspect the composed configuration without running it:
# python -m ccflow.examples.calculator.explain function=power

if __name__ == "__main__":
    main()
