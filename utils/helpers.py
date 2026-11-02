from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def defaults() -> dict:
    with (ROOT / 'config/settings.yaml').open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)
