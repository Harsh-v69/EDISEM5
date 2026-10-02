"""The fixed herb/drug scope for the whole project (spec section 8: do not expand later)."""
from pathlib import Path

import yaml


def load_scope(path="config/scope.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
