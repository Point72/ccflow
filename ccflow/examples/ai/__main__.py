import hydra

from ccflow.utils.hydra import cfg_run

__all__ = ("main",)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg):
    cfg_run(cfg)


# Run the whole graph offline, with no credentials. TestModel generates placeholder text that cannot
# satisfy a real-quotation check, hence checks=off:
# python -m ccflow.examples.ai checks=off
#
# Run it against a real model, with the mechanical constraints enforced:
# python -m ccflow.examples.ai model=openai publisher=print
#
# Swap one axis at a time; each is a config group:
# python -m ccflow.examples.ai spec=haiku
# python -m ccflow.examples.ai rubrics=strict
# python -m ccflow.examples.ai profile=steelman
# python -m ccflow.examples.ai lessons=house_style
# python -m ccflow.examples.ai agents=poem_concise
#
# Judge both presentation orders and report only what survives both:
# python -m ccflow.examples.ai task=counterbalanced
#
# Let each advocate answer the other before the arbiter sees anything:
# python -m ccflow.examples.ai task=rebuttal
# python -m ccflow.examples.ai task=rebuttal_counterbalanced
#
# Write each stage to disk and chain the single-node tasks, which pins the poem so the same two
# cases can be re-judged under a different model:
# python -m ccflow.examples.ai task=produce publisher=json
# python -m ccflow.examples.ai task=advocate publisher=json position=pro
# python -m ccflow.examples.ai task=advocate publisher=json position=con
# python -m ccflow.examples.ai task=arbitrate publisher=json model=anthropic

if __name__ == "__main__":
    main()
