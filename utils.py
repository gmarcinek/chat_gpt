# utils.py
# Zbiór prostych funkcji pomocniczych (IO, daty, YAML/JSON, PDF, szablony).

from __future__ import annotations
import os
import json
import time
import yaml
import pathlib
from typing import Dict, Any


# ===== Czas / ID procesu =====
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_proc_id() -> str:
    # yyMMdd-HHmmss (np. 251109-003015)
    return time.strftime("%y%m%d-%H%M%S")


# ===== Pliki / katalogi =====
def ensure_dir_for(path: str):
    """Utwórz katalog nadrzędny dla pliku, jeśli nie istnieje."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


# ===== YAML / JSON =====
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, data: Any):
    ensure_dir_for(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===== Czytanie tekstu z pliku (TXT/MD/JSON/PDF) =====
def _read_pdf_to_text(path: str) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise RuntimeError("Brak PyPDF2. Zainstaluj: pip install PyPDF2")

    parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            parts.append(f"\n\n[[PAGE {i+1}]]\n{page_text}")
    return "".join(parts).strip()


def read_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md", ".json"}:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        return _read_pdf_to_text(path)
    raise RuntimeError(f"Nieobsługiwane rozszerzenie: {ext}")


# ===== Najprostszy renderer kluczy {key} =====
def simple_render(template: str, ctx: Dict[str, Any]) -> str:
    out = template
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# ===== Zapis wyniku kroku =====
def save_result_with_prefix(base_prefix: str, proc_id: str, filename: str, content: str) -> str:
    """
    Zapisuje wynik kroku do JSON.
    - Jeśli 'content' to poprawny JSON → zapisuje ten obiekt.
    - W przeciwnym razie: {"raw_output": "..."}.
    Zwraca pełną ścieżkę pliku.
    """
    out_dir = base_prefix.rstrip("/\\")
    out_path = f"{out_dir}/{proc_id}_{filename}"
    ensure_dir_for(out_path)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_output": content}
    save_json(out_path, parsed)
    return out_path
