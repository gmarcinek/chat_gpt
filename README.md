# Runner zadań LLM (YAML)

Ten projekt uruchamia wieloetapowe zadania LLM zdefiniowane w pliku YAML. Modele i providerzy są konfigurowani w `models.yaml`, a przebieg zadań (prompty, wejścia, kroki) w `config.yaml`.

## Wymagania

- Python 3.10+
- Klucze API (zależnie od wybranego providera):
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Ollama: lokalny runtime (opcjonalnie `OLLAMA_BASE_URL`)
- Pakiety: `python-dotenv`, `PyYAML`, `requests`, oraz (w razie potrzeby) `openai`, `anthropic`

## Instalacja

Windows (PowerShell):

```powershell
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install python-dotenv pyyaml requests openai anthropic
```

macOS/Linux (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install python-dotenv pyyaml requests openai anthropic
```

## Klucze i plik `.env`

- Skopiuj `./.env.example` do `./.env` i uzupełnij wartości:
  - `OPENAI_API_KEY=...`
  - `ANTHROPIC_API_KEY=...`
  - (opcjonalnie) `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- Plik `.env` jest ładowany automatycznie (`python-dotenv`).

## Konfiguracja modeli — `models.yaml`

- Sekcja `providers`: aliasy providerów (`openai`, `anthropic`, `ollama`).
- Sekcja `models`: modele wraz z atrybutami:
  - `provider` — dostawca dla danego modelu,
  - `input_context` — limit kontekstu wejściowego,
  - `max_output` — domyślny limit tokenów wyjścia,
  - `vision` — czy model wspiera obraz.
- Sekcja `defaults`: wartości domyślne, gdy nie określono przy modelu.

Klient LLM (`llm_client.py`) sam dobiera providera i limity na podstawie nazwy modelu. Jeśli w wywołaniu nie podasz parametrów (`temperature`, `max_output_tokens`), zostaną użyte domyślne wartości z `models.yaml` (oraz `temperature=0.0`).

## Konfiguracja uruchomienia — `config.yaml`

Kluczowe pola:

- `model` — opcjonalne; jeśli brak, klient wybierze domyślny model z `models.yaml`.
- `result_output_prefix` — katalog wyjściowy (np. `output/ner_results`).
- `system_prompt_ref` — klucz do sekcji `prompts` w `config.yaml`.
- `prompts` — słownik promptów (np. `SYSTEM_PROMPT`, `__INITIAL_USER_PROMPT__`, itd.).
- `inputs.source_pdf` — ścieżka do dokumentu (np. `./data/owu.pdf`).
- `tasks` — lista kroków. Każdy krok może mieć:
  - `id`, `role` (np. `user`),
  - `prompt_ref` lub bezpośrednio `prompt`,
  - opcjonalnie `render` (proste podstawienia),
  - `save_as` (nazwa pliku wynikowego dla kroku).

Specjalnie: gdy `prompt_ref: __INITIAL_USER_PROMPT__`, zawartość wskazanego PDF zostanie wstrzyknięta do rozmowy.

## Uruchamianie

Podstawowe polecenie:

```bash
python run_tasks.py -c config.yaml
```

Argument `-c/--config` wskazuje plik konfiguracyjny (domyślnie `config.yaml`).

W trakcie działania narzędzie:

- Ładuje `config.yaml` i buduje historię rozmowy `messages` według `tasks`.
- Wywołuje LLM przez zunifikowany `LLMClient`, który:
  - wybiera providera z `models.yaml`,
  - ustawia domyślny `max_output_tokens` z `models.yaml`,
  - używa `temperature=0.0`, o ile nie podano inaczej.

## Wyniki

- Każdy krok zapisuje rezultat do JSON w katalogu `result_output_prefix`:
  - nazwa pliku: `PROC_ID_nazwa_kroku.json`.
- Pełna historia rozmowy zapisywana jest jako jeden plik `PROC_ID_conversation.json` w tym samym katalogu.

## Zmiana dostawcy lub modelu

W `config.yaml` ustaw `model` na dowolny zdefiniowany w `models.yaml`, np.:

- OpenAI: `model: gpt-4.1`
- Anthropic: `model: claude-3.5-sonnet`
- Ollama: `model: qwen2.5-coder`

Dla Ollama upewnij się, że model jest dostępny lokalnie (np. `ollama pull qwen2.5-coder`) i usługa działa (`OLLAMA_BASE_URL` lub domyślne `http://127.0.0.1:11434`).

## Rozwiązywanie problemów

- Brak pakietu OpenAI/Anthropic: `pip install openai` lub `pip install anthropic`.
- Brak kluczy API: uzupełnij `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` w `.env`.
- Błąd połączenia z Ollama: sprawdź, czy działa demon Ollama i port; ustaw `OLLAMA_BASE_URL` gdy niestandardowy.
- Dziwne znaki w logach konsoli Windows: ustaw UTF-8 (`chcp 65001`) — nie wpływa na działanie.

## Pliki kluczowe

- `run_tasks.py` — entrypoint CLI.
- `llm_client.py` — zunifikowany klient LLM; czyta `models.yaml` i stosuje domyślne limity.
- `models.yaml` — definicje modeli i providerów.
- `config.yaml` — workflow: prompty, wejścia, zadania.
- `utils.py` — pomocnicze funkcje I/O i renderowanie.

## Szybki start (minimalny config)

Przykładowy minimalny `config.yaml` z dwoma krokami:

```yaml
model: gpt-4.1
result_output_prefix: output/ner_results
prompts:
  SYSTEM_PROMPT: |
    Jesteś pomocnym asystentem.
  __INITIAL_USER_PROMPT__: |
    Oto plik "{pdf_basename}". Przeczytaj i czekaj na wytyczne.
inputs:
  source_pdf: ./data/owu.pdf
system_prompt_ref: SYSTEM_PROMPT
tasks:
  - id: init
    role: user
    prompt_ref: __INITIAL_USER_PROMPT__
    render:
      pdf_basename: "{inputs.source_pdf|basename}"
    save_as: 0.init.json
  - id: question
    role: user
    prompt: |
      Wypisz 3 najważniejsze pojęcia z dokumentu.
    save_as: 1.summary.json
```

Uruchom:

```bash
python run_tasks.py -c config.yaml
```
