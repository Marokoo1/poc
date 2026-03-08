from pathlib import Path
import yaml


def load_settings(settings_path: str = "config/settings.yaml") -> dict:
    path = Path(settings_path)

    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
