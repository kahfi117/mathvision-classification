import os
import json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)