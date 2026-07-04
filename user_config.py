"""
user_config.py — Tiny persisted-settings store (survives app restarts).

Currently only stores the GGUF model folder path, so the user only has to
type/scan it once instead of every time the app starts.
"""

import json
from pathlib import Path

USER_CONFIG_PATH = Path(__file__).parent / "user_config.json"


def load_user_config() -> dict:
    try:
        if USER_CONFIG_PATH.exists():
            return json.loads(USER_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Config] Could not read '{USER_CONFIG_PATH}': {e}")
    return {}


def save_user_config(updates: dict) -> None:
    try:
        data = load_user_config()
        data.update(updates)
        USER_CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Config] Could not write '{USER_CONFIG_PATH}': {e}")


# Loaded once at import time; individual values are re-read via
# load_user_config() when freshness matters (there's only ever one process
# writing this file, so a module-level snapshot is fine for startup use).
USER_CONFIG = load_user_config()
