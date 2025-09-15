import yaml, os

def load_config(path: str = "config/config.yaml") -> dict:
    import yaml  # lazy import so training doesn't need PyYAML
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
