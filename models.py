# models.py
# Wersja korzystająca z konfiguracji YAML zamiast stałych w kodzie.

import yaml
from enum import Enum
from typing import Dict, Any
from pathlib import Path


# === 1. Enum dla providerów ===
class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# === 2. Klasa Models (symboliczna – może być zachowana dla kompatybilności) ===
class Models:
    pass


# === 3. Ładowanie konfiguracji YAML ===
def load_model_config(yaml_path: str = "models.yaml") -> Dict[str, Any]:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku konfiguracyjnego: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Nadpisanie brakujących sekcji domyślnych pustymi
    cfg.setdefault("defaults", {})
    cfg.setdefault("models", {})
    cfg.setdefault("providers", {})

    return cfg


# === 4. Główna konfiguracja ===
CONFIG = load_model_config()


# === 5. Funkcje API ===
def resolve_model(model_identifier: str) -> str:
    """
    Przyjmuje:
      - nazwę stałej z klasy Models (np. 'GPT_4_1'), albo
      - literal nazwy modelu API (np. 'gpt-4.1').
    Zwraca literal nazwy modelu API.
    """
    items = {k: v for k, v in vars(Models).items() if not k.startswith("_")}
    if not model_identifier:
        return next(iter(CONFIG["models"].keys()), "gpt-4.1")

    if model_identifier in items.values():
        return model_identifier
    if model_identifier in items:
        return items[model_identifier]
    return model_identifier


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Zwraca pełną konfigurację dla danego modelu,
    z uwzględnieniem wartości domyślnych.
    """
    defaults = CONFIG.get("defaults", {})
    model_cfg = CONFIG.get("models", {}).get(model_name, {})

    merged = {
        "provider": model_cfg.get("provider", defaults.get("provider", "openai")),
        "input_context": model_cfg.get("input_context", defaults.get("input_context", 32768)),
        "max_output": model_cfg.get("max_output", defaults.get("max_output", 8192)),
        "vision": model_cfg.get("vision", defaults.get("vision", False)),
    }
    return merged


def get_model_provider(model_name: str) -> ModelProvider:
    provider_str = get_model_config(model_name)["provider"]
    try:
        return ModelProvider(provider_str)
    except ValueError:
        return ModelProvider.OPENAI


def get_model_input_limit(model_name: str) -> int:
    return get_model_config(model_name)["input_context"]


def get_model_output_limit(model_name: str) -> int:
    return get_model_config(model_name)["max_output"]


def supports_vision(model_name: str) -> bool:
    return get_model_config(model_name)["vision"]


# === 6. (Opcjonalnie) Inicjalizacja klasy Models symbolicznie ===
def _populate_models_class():
    """Automatycznie dodaje do klasy Models wszystkie nazwy modeli jako atrybuty."""
    for name in CONFIG["models"].keys():
        const_name = name.upper().replace("-", "_").replace(":", "_")
        setattr(Models, const_name, name)


_populate_models_class()


# === 7. Test manualny ===
if __name__ == "__main__":
    for m in ["gpt-5", "claude-3.5-sonnet", "llama3.2-vision:11b", "nieistniejący"]:
        cfg = get_model_config(m)
        print(f"\nModel: {m}")
        print(f"  Provider: {cfg['provider']}")
        print(f"  Input: {cfg['input_context']}")
        print(f"  Output: {cfg['max_output']}")
        print(f"  Vision: {cfg['vision']}")
