from ccflow.utils.hydra import cfg_explain_cli

from .__main__ import main


def explain():
    cfg_explain_cli(config_path="config", config_name="base", hydra_main=main)


if __name__ == "__main__":
    explain()
