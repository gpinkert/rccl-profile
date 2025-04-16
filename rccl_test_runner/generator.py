from copy import deepcopy
import yaml

# Define the environment variables and their test values
env_var_values = {
    "NCCL_MIN_NCHANNELS": [12, 16, 32],
    "NCCL_MAX_NCHANNELS": [1, 4, 8, 16, 32, 64, 128],
    "NCCL_NCHANNELS_PER_NET_PEER": [1, 2, 4, 8, 16, 32],
    "NCCL_NCHANNELS_PER_PEER": [1, 2, 4, 8, 16, 32],
    "NCCL_MIN_P2P_NCHANNELS": [1, 2, 4, 8, 16],
    "NCCL_MAX_P2P_NCHANNELS": [4, 8, 16, 32],
    "NCCL_IGNORE_CPU_AFFINITY": [0, 1],
    "NCCL_MAX_CTAS": [1, 2, 4, 8],
    "NCCL_MIN_CTAS": [1, 2, 4, 8]
}
all_keys = list(env_var_values.keys())

# Base configuration template
base_config = {
    "collectives": ["all"],
    "start_size": 1,
    "end_size": "16g",
    "iterations": 2,
    "operation": ["all"],
    "step_details": [{"type": "multiple", "value": 2}],
    "datatypes": ["all"]
}

# Custom class to force inline (flow) formatting for specific lists
class InlineList(list):
    pass

def inline_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(InlineList, inline_list_representer)

# Build the configurations
final_configs = {}

# Create an entry for each env var value
for var, values in env_var_values.items():
    for val in values:
        name = f"{var.lower()}_{val}"
        config = deepcopy(base_config)
        # Force inline representation for these keys
        config["collectives"] = InlineList(config["collectives"])
        config["operation"]   = InlineList(config["operation"])
        config["datatypes"]   = InlineList(config["datatypes"])
        config["ENV_VARS"] = []
        # Set the target variable to its test value
        config["ENV_VARS"].append({"id": var, "value": val})
        # Set all other env vars to 0
        for other in all_keys:
            if other != var:
                config["ENV_VARS"].append({"id": other, "value": "default"})
        final_configs[name] = config

# Add an entry with every env var turned off
off_entry = deepcopy(base_config)
off_entry["collectives"] = InlineList(off_entry["collectives"])
off_entry["operation"]   = InlineList(off_entry["operation"])
off_entry["datatypes"]   = InlineList(off_entry["datatypes"])
off_entry["ENV_VARS"] = [{"id": key, "value": 0} for key in all_keys]
final_configs["all_nccl_env_vars_off"] = off_entry

# Dump the YAML (without anchors)
yaml_output = yaml.dump(final_configs, sort_keys=False)

# Save the YAML file
final_yaml_path = "./nccl_env_var_sweep_individual.yaml"
with open(final_yaml_path, "w") as f:
    f.write(yaml_output)

print(final_yaml_path)
