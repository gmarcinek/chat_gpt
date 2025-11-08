# run_tasks.py
# Wykonuje DOWOLNĄ liczbę kroków z YAML (nowa struktura), zapisuje progres przyrostowo.
# ZMIANA: dokument PDF jest dołączany JEDNORAZOWO na początku rozmowy, wraz z promptem "__INITIAL_USER_PROMPT__".
# Kolejne kroki odwołują się do już wprowadzonej treści (bez ponownego wstrzykiwania dokumentu).

import os
import json
import time
import yaml
import pathlib
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

# ====== OpenAI (Chat Completions API) ======
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"))

# ================= Pomocnicze =================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def make_proc_id() -> str:
    # Krótki znacznik czasu do prefixu plików, np. 251108-221045
    return time.strftime("%y%m%d-%H%M%S")

def ensure_dir_for(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path: str, data: Any):
    ensure_dir_for(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".json"]:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
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
    raise RuntimeError(f"Nieobsługiwane rozszerzenie: {ext}")

def simple_render(template: str, ctx: Dict[str, Any]) -> str:
    out = template
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v))
    return out

def llm_call(model: str, temperature: float, messages: List[Dict[str, str]],
             seed: Optional[int]=None, max_output_tokens: Optional[int]=None) -> str:
    """
    Używa Chat Completions API (messages=...).
    'seed' jest ignorowany (feature Responses API). max_output_tokens mapowany na max_tokens.
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens

    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    return text

def save_result_with_prefix(base_prefix: str, proc_id: str, filename: str, content: str) -> str:
    # zapisuje JSON pod ścieżką: "<base_prefix>/<proc_id>_<filename>"
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
        raise FileNotFoundError("Brak config.yaml")

    cfg = load_yaml(config_path)

    model = cfg.get("model", "gpt-4.1")
    temperature = float(cfg.get("temperature", 0.0))
    seed = cfg.get("seed", None)  # ignorowane w chat.completions
    max_output_tokens = cfg.get("max_output_tokens", None)

    # Nowa struktura YAML
    prompts = cfg.get("prompts", {}) or {}
    system_prompt_ref = cfg.get("system_prompt_ref", None)
    system_prompt = prompts.get(system_prompt_ref, "").strip() if system_prompt_ref else ""

    inputs = cfg.get("inputs", {}) or {}
    source_pdf = inputs.get("source_pdf", "")
    result_output_prefix = cfg.get("result_output_prefix", "output/ner_results")
    
    tasks = cfg.get("tasks", []) or []
    if not tasks:
        raise RuntimeError("Brak sekcji 'tasks' w YAML.")

    # proc_id i nagłówek logów - PRZENIEŚ TUTAJ
    proc_id = make_proc_id()
    conversation_path = f"{result_output_prefix.rstrip('/\\')}/{proc_id}_conversation.json"
    
    step_ids = [t.get("id", f"task_{i+1}") for i, t in enumerate(tasks)]
    print(f"🚀 Start procesu {proc_id}")
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

    # Historia rozmowy – dopisywana przyrostowo
    history_log: List[Dict[str, Any]] = load_json(conversation_path, default=[])

    # Bieżące messages dla rozmowy z modelem
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # === JEDNORAZOWE wstrzyknięcie: prompt analizujący PDF + treść dokumentu ===
    initial_pdf_prompt = prompts.get("__INITIAL_USER_PROMPT__", "").strip()
    if doc_text and initial_pdf_prompt:
        initial_user = simple_render(initial_pdf_prompt, render_ctx)
        initial_payload = f"{initial_user}\n\n=== DOKUMENT_START ===\n{doc_text}\n=== DOKUMENT_KONIEC ==="
        messages.append({"role": "user", "content": initial_payload})
        print("📎 Dołączono dokument do historii (jednorazowo).")

    previous_assistant_text: Optional[str] = None

    total = len(tasks)
    done = 0

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("id", f"task_{idx}")
        role = task.get("role", "user")
        save_as = task.get("save_as", f"{task_id}.json")
        prompt_text = task.get("prompt", None)

        # prompt_ref → prompts[ref]
        if prompt_text is None:
            ref = task.get("prompt_ref", None)
            if ref:
                if ref not in prompts:
                    raise RuntimeError(f"Brak promptu '{ref}' w sekcji prompts.")
                prompt_text = prompts[ref]
            else:
                raise RuntimeError(f"Task '{task_id}' nie ma 'prompt' ani 'prompt_ref'.")

        # render (prosta templatyzacja wartości)
        if "render" in task:
            for k, v in (task["render"] or {}).items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    key = v[1:-1]
                    render_ctx[k] = render_ctx.get(key, "")
                else:
                    render_ctx[k] = v
        final_prompt = simple_render(prompt_text, render_ctx)

        # log – start kroku
        print(f"▶️  [{idx}/{total}] {task_id} — uruchamianie…")

        # opcja: dołącz poprzednią odpowiedź jako assistant
        if task.get("append_previous_assistant", False) and previous_assistant_text:
            messages.append({"role": "assistant", "content": previous_assistant_text})

        # UWAGA: NIE DOŁĄCZAMY już dokumentu – jest w historii od startu
        messages.append({"role": role, "content": final_prompt})

        # Wywołanie LLM
        try:
            response_text = llm_call(
                model=model,
                temperature=temperature,
                messages=messages,
                seed=seed,
                max_output_tokens=max_output_tokens
            )
        except Exception as e:
            response_text = json.dumps({"error": str(e)}, ensure_ascii=False)
            print(f"❌  [{idx}/{total}] {task_id} — błąd: {e}")

        # Zapis wyniku (z prefixem procesu)
        out_path = save_result_with_prefix(result_output_prefix, proc_id, save_as, response_text)

        # Zapis przyrostowy do historii
        record = {
            "timestamp": now_iso(),
            "process_id": proc_id,
            "task_index": idx,
            "task_id": task_id,
            "model": model,
            "temperature": temperature,
            "messages": messages,          # dokładnie to, co widzi LLM
            "result_file": out_path
        }
        history_log.append(record)
        save_json(conversation_path, history_log)

        # Uaktualnij rozmowę i licznik
        previous_assistant_text = response_text
        messages.append({"role": "assistant", "content": response_text})
        done += 1

        # log – koniec kroku
        print(f"✅  [{idx}/{total}] {task_id} — zapisano: {out_path}")
        print(f"📈 Postęp: {done}/{total} zakończonych\n")

    print(f"🎉 Zakończono proces {proc_id} — łączna liczba kroków: {total}")
    print(f"🗂️ Historia konwersacji: {conversation_path}")

# ================= Uruchomienie =================
if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_from_yaml(cfg_path)