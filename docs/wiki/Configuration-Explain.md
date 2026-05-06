# Tracing Hydra Config Composition

Ever stared at a resolved Hydra config and wondered: *where did that value come from?* Which YAML file set it, which override changed it, and in what order did composition happen? This guide shows you how to answer those questions with CCRT's Config Explain app.

By working through the examples below you will learn to:
- Dump a full config explanation to a text file on the CLI
- Launch an interactive browser UI that lets you explore variable origins visually

## Prerequisites

If you are using a CCRT conda environment, `ccflow` is already installed. Otherwise install it:

```bash
pip install ccflow
```

## CLI: Export Config Explanation to a File

Run the following to produce a text explanation of the composed config:

```bash
python -m ccflow.utils.hydra --config-path /PATH/TO/CONFIG_DIR --config-name CONFIG_FILE --no-gui > OUTPUT_FILE

# E.g. if my main conf file is named config/base.yaml    
python -m ccflow.utils.hydra \
    --config-path /home/steve/oss/Point72/ccflow/ccflow/examples/etl/config \
    --config-name base \
    --no-gui > conf.explain.txt
    
# Look at pretty print of the conf
less conf.explain.txt
```

Open `conf.explain.txt` to see every variable, its resolved value, and where it originated.

## GUI: Interactive Browser UI

To launch the explanation UI on a port you can reach from your browser:

```bash
python -m ccflow.utils.hydra --address IP_ADDR --port PORT --config-path /PATH/TO/CONFIG_DIR --config-name CONFIG_FILE

# e.g. I want a UI so I tell the CCRT's Config Explain app to run as a server (--address 0.0.0.0) on port 5555   

python -m ccflow.utils.hydra \
    --address 0.0.0.0 \
    --port 5555 \
    --config-path /home/steve/oss/Point72/ccflow/ccflow/examples/etl/config \
    --config-name base
```

Then open one's browser and navigate to `http://<YOUR-HOST>:5555` to explore the config tree interactively.
Replace **YOUR-HOST** with the Linux host where you are running the config explain app.

E.g. of YAML and what it looks like in the Interactive Browser UI
![Config YAML](https://raw.githubusercontent.com/point72/ccflow/main/docs/img/wiki/config-explain/ccflow-yaml.png)
![Config YAML in the Interactive Browser UI](https://raw.githubusercontent.com/point72/ccflow/main/docs/img/wiki/config-explain/ccflow-ui.png)
