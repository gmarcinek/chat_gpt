# run_tasks.py
import os
import json
import time
import yaml
import pathlib
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# === zunifikowany klient LLM ===
from llm_client import LLMClient

# === utils ===
from utils import (
    now_iso,
    make_proc_id,
    ensure_dir_for,
    load_yaml,
    save_json,
    read_text_from_path,
    simple_render,
    save_result_with_prefix,
)

# === NOWE: globalny klient
_llm = LLMClient()

def llm_call(messages: List[Dict[str, str]],
             model: Optional[str]=None,
             temperature: Optional[float]=None,
             seed: Optional[int]=None,
             max_output_tokens: Optional[int]=None) -> str:
    return _llm.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )

def save_result_with_prefix(base_prefix: str, proc_id: str, filename: str, content: str) -> str:
    out_dir = base_prefix.rstrip("/\\")
    out_path = f"{out_dir}/{proc_id}_{filename}"
    ensure_dir_for(out_path)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_output": content}
    save_json(out_path, parsed)
    return out_path

# ================= Główna funkcja =================
def run_from_yaml(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Brak pliku konfiguracji: {config_path}")

    cfg = load_yaml(config_path)

    model = cfg.get("model", None)
    temperature = None
    seed = None
    max_output_tokens = None

    prompts = cfg.get("prompts", {}) or {}
    system_prompt_ref = cfg.get("system_prompt_ref", None)
    system_prompt = prompts.get(system_prompt_ref, "").strip() if system_prompt_ref else ""

    inputs = cfg.get("inputs", {}) or {}
    source_pdf = inputs.get("source_pdf", "")
    result_output_prefix = cfg.get("result_output_prefix", "output/ner_results")
    
    tasks = cfg.get("tasks", []) or []
    if not tasks:
        raise RuntimeError("Brak sekcji 'tasks' w YAML.")

    proc_id = make_proc_id()
    conversation_path = f"{result_output_prefix.rstrip('/\\')}/{proc_id}_conversation.json"
    
    step_ids = [t.get("id", f"task_{i+1}") for i, t in enumerate(tasks)]
    print(f"🚀 Start procesu {proc_id}")
    print(f"📄 Config: {config_path}")
    print(f"🧩 Kroków: {len(tasks)} → {', '.join(step_ids)}")
    print(f"🧠 Model: {model} · 🌡️ T={temperature}\n")

    # Przygotuj treść dokumentu (jeśli jest)
    doc_text = ""
    if source_pdf and os.path.exists(source_pdf):
        try:
            doc_text = read_text_from_path(source_pdf)
        except Exception as e:
            print(f"⚠️ Nie udało się odczytać dokumentu: {e}")

    # Kontekst renderowania
    render_ctx = {
        "pdf_basename": os.path.basename(source_pdf) if source_pdf else "",
        "inputs.source_pdf|basename": os.path.basename(source_pdf) if source_pdf else "",
    }

    # Bieżące messages dla rozmowy z modelem
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    previous_assistant_text: Optional[str] = None

    total = len(tasks)
    done = 0

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("id", f"task_{idx}")
        role = task.get("role", "user")
        save_as = task.get("save_as", f"{task_id}.json")
        prompt_text = task.get("prompt", None)

        if prompt_text is None:
            ref = task.get("prompt_ref", None)
            if ref:
                if ref not in prompts:
                    raise RuntimeError(f"Brak promptu '{ref}' w sekcji prompts.")
                prompt_text = prompts[ref]
            else:
                raise RuntimeError(f"Task '{task_id}' nie ma 'prompt' ani 'prompt_ref'.")

        # render
        if "render" in task:
            for k, v in (task["render"] or {}).items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    key = v[1:-1]
                    render_ctx[k] = render_ctx.get(key, "")
                else:
                    render_ctx[k] = v
        final_prompt = simple_render(prompt_text, render_ctx)

        print(f"▶️  [{idx}/{total}] {task_id} — uruchamianie…")

        if task.get("append_previous_assistant", False) and previous_assistant_text:
            messages.append({"role": "assistant", "content": previous_assistant_text})

        # wstrzyknięcie dokumentu dla __INITIAL_USER_PROMPT__
        prompt_ref = task.get("prompt_ref", None)
        if prompt_ref == "__INITIAL_USER_PROMPT__" and doc_text:
            payload = f"{final_prompt}\n\n=== DOKUMENT_START ===\n{doc_text}\n=== DOKUMENT_KONIEC ==="
            messages.append({"role": role, "content": payload})
            print(f"📎 Dołączono dokument do taska {task_id} (prompt_ref: {prompt_ref})")
        else:
            messages.append({"role": role, "content": final_prompt})

        # Wywołanie LLM
        try:
            response_text = llm_call(
                messages=messages,
                model=model,
            )
        except Exception as e:
            response_text = json.dumps({"error": str(e)}, ensure_ascii=False)
            print(f"❌  [{idx}/{total}] {task_id} — błąd: {e}")

        # Zapis wyniku
        out_path = save_result_with_prefix(result_output_prefix, proc_id, save_as, response_text)

        # Dopisz odpowiedź assistanta
        messages.append({"role": "assistant", "content": response_text})

        # Zapisz stan rozmowy
        save_json(conversation_path, messages)

        previous_assistant_text = response_text
        done += 1

        print(f"✅  [{idx}/{total}] {task_id} — zapisano: {out_path}")
        print(f"📈 Postęp: {done}/{total} zakończonych\n")

    print(f"🎉 Zakończono proces {proc_id} — łączna liczba kroków: {total}")
    print(f"🗂️ Historia konwersacji: {conversation_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uruchom zadania z pliku konfiguracyjnego YAML"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Ścieżka do pliku konfiguracyjnego (domyślnie: config.yaml)"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Używam konfiguracji: {args.config}\n")
    run_from_yaml(args.config)