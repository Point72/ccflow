# Tracing Hydra Config Composition

As configurations grow in complexity, it becomes important to trace the origin of each resolved value: which YAML file defined it, which override modified it, and in what order composition occurred. The Config Explain utility provides both a CLI and an interactive browser interface for inspecting the full provenance of a composed Hydra configuration.

Additionally, a synthetic `__options__` element is injected into each config group node to display the other available config group options that could be selected in place of the current one.

> **Note:** A *config group* is a subdirectory under the Hydra config root that represents a swappable slot in the configuration. Each YAML file within that directory is a *config group option* — a named alternative that can be selected via the `defaults:` list or a CLI override (e.g. `+group=option`). The Config Explain UI surfaces these relationships so that users can see not only the active selection but also what alternatives exist. 
```
Directory Layout

config/
    base.yaml              # root config
    extract/
        rest.yaml          # option "rest" for group "extract"
    transform/
        links.yaml         # option "links" for group "transform"
    load/
        db.yaml            # option "db" for group "load"

Root Config (base.yaml)
```

The utility supports two modes of operation:

- Exporting a full config explanation to a text file via the CLI
- Launching an interactive browser UI for visual exploration of variable origins

## Prerequisites

Install ccflow:

```bash
pip install ccflow
```

## CLI: Export Config Explanation to a File

The following command produces a text explanation of the composed config:

```bash
python -m ccflow.utils.hydra --config-path /PATH/TO/CONFIG_DIR --config-name CONFIG_FILE --no-gui > OUTPUT_FILE

# Example: the root config file is config/base.yaml
python -m ccflow.utils.hydra \
    --config-path /home/user/src/ccflow/ccflow/examples/etl/config \
    --config-name base \
    --no-gui > conf.explain.txt
    
# View the output
less conf.explain.txt
```

Open `conf.explain.txt` to see every variable, its resolved value, and where it originated.

## GUI: Interactive Browser UI

To launch the explanation UI as an HTTP server:

```bash
python -m ccflow.utils.hydra --address IP_ADDR --port PORT --config-path /PATH/TO/CONFIG_DIR --config-name CONFIG_FILE

# Example: serve the UI on all interfaces, port 5555
python -m ccflow.utils.hydra \
    --address 0.0.0.0 \
    --port 5555 \
    --config-path /home/user/src/ccflow/ccflow/examples/etl/config \
    --config-name base
```

Navigate to `http://HOST:PORT` in a browser to explore the config tree interactively, where **HOST** and **PORT** correspond to the address and port specified above.

The following screenshots show a sample YAML configuration and its representation in the interactive UI:
![Config YAML](https://raw.githubusercontent.com/point72/ccflow/main/docs/img/wiki/config-explain/ccflow-yaml.options.png)
![Config YAML in the Interactive Browser UI](https://raw.githubusercontent.com/point72/ccflow/main/docs/img/wiki/config-explain/ccflow-ui.options.png)
