from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return _PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path from config.yaml to an absolute Path."""
    return _PROJECT_ROOT / relative_path


if __name__ == "__main__":
    cfg = load_config()
    print(f"Classes ({len(cfg['classes'])}): {cfg['classes']}")
    print(f"Architecture : {cfg['model']['architecture']}")
    print(f"Project root : {get_project_root()}")
