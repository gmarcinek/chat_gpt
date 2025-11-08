import os, json, time, yaml
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from models import Models, get_model_provider

# ---------------- helpers ----------------
def resolve_model(name_from_yaml: str) -> str:
    if not name_from_yaml:
        return Models.GPT_4_1
    all_vals = {v for k, v in vars(Models).items() if not k.startswith("_")}
    if name_from_yaml in all_vals:
        return name_from_yaml
    all_enum = {k: v for k, v in vars(Models).items() if not k.startswith("_")}
    if name_from_yaml in all_enum:
        return all_enum[name_from_yaml]
    return name_from_yaml

def render_prompt(tpl: str, **kw) -> str:
    try:
        return tpl.format(**kw)
    except KeyError as e:
        raise RuntimeError(f"Brak wartości dla placeholdera {e} w pdf_user_prompt")

def is_gpt5(model_name: str) -> bool:
    return model_name.lower().startswith("gpt-5")

def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_result_file(prefix: str, content: str) -> str:
    out_dir = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    os.makedirs(out_dir, exist_ok=True)
    i = 1
    while True:
        out_path = os.path.join(out_dir, f"{base}_{i:03d}.json")
        if not os.path.exists(out_path):
            break
        i += 1
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_output": content}
    save_json(out_path, parsed)
    print(f"[Zapisano wynik do pliku: {out_path}]")
    return out_path

# ---------------- config ----------------
CONFIG = "config.yaml"
if not os.path.exists(CONFIG):
    raise FileNotFoundError("Brak config.yaml")

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = resolve_model(cfg.get("model"))
SYSTEM_PROMPT = cfg.get("system_prompt", "")
PDF_USER_PROMPT_TPL = cfg.get("pdf_user_prompt", "")
SOURCE_PDF = cfg.get("source_pdf")
RESULT_PREFIX = cfg.get("result_output_prefix", "output/result")
TEMPERATURE = float(cfg.get("temperature", 0.0))

if not SYSTEM_PROMPT:
    raise RuntimeError("Brak 'system_prompt' w config.yaml")
if not PDF_USER_PROMPT_TPL:
    raise RuntimeError("Brak 'pdf_user_prompt' w config.yaml")
if not SOURCE_PDF or not os.path.exists(SOURCE_PDF):
    raise FileNotFoundError(f"Nie znaleziono PDF: {SOURCE_PDF}")
if get_model_provider(MODEL_NAME).value != "openai":
    raise RuntimeError("Ten skrypt obsługuje tylko modele OpenAI.")

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not api_key:
    raise RuntimeError("Brak OPENAI_API_KEY / OPENAI_KEY")

client = OpenAI(api_key=api_key)

# 1) Upload PDF
with open(SOURCE_PDF, "rb") as f:
    uploaded = client.files.create(file=f, purpose="assistants")
file_id = uploaded.id
print(f"[Wgrano PDF] file_id={file_id}")

# 2) Przygotuj prompt
pdf_basename = os.path.basename(SOURCE_PDF)
user_prompt = render_prompt(PDF_USER_PROMPT_TPL, pdf_basename=pdf_basename)

# 3) Wywołaj nowe Responses API z file_search
response_args = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt,
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [{"type": "file_search"}]
                }
            ]
        }
    ],
    "tools": [{"type": "file_search"}]
}

if not is_gpt5(MODEL_NAME):
    response_args["temperature"] = TEMPERATURE

response = client.chat.completions.create(**response_args)

# 4) Pobierz odpowiedź
assistant_text = response.choices[0].message.content

if not assistant_text:
    raise RuntimeError("Brak treści odpowiedzi.")

print("=== OUTPUT ===")
print(assistant_text)

# 5) Zapis do pliku
save_result_file(RESULT_PREFIX, assistant_text)