from enum import Enum
from typing import Dict

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class Models:
    # OpenAI GPT-5 — API: 272k input + 128k reasoning/output = 400k total
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"

    # OpenAI GPT-4.1 — 1,047,576 input, 32,768 output
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

    # OpenAI GPT-4o (legacy family) — 128k input, 16,384 output
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Anthropic
    CLAUDE_4_SONNET = "claude-sonnet-4"
    CLAUDE_4_OPUS = "claude-opus-4"
    CLAUDE_4_1_OPUS = "claude-opus-4.1"           # nowszy snapshot
    CLAUDE_3_7_SONNET = "claude-3.7-sonnet"
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3.5-haiku"
    CLAUDE_3_HAIKU = "claude-3-haiku"

    # Ollama – coding
    QWEN_CODER = "qwen2.5-coder"
    QWEN_CODER_32B = "qwen2.5-coder:32b"
    CODESTRAL = "codestral"              # Codestral 25.01/25.08 (Mistral)

    # Ollama – vision
    LLAMA_VISION_11B = "llama3.2-vision:11b"
    LLAMA_VISION_90B = "llama3.2-vision:90b"
    QWEN_VISION_7B = "qwen2.5vl:7b"
    GEMMA3_12B = "gemma3:12b"            # Gemma 3 jest multimodalna
    # (opcjonalnie: GEMMA3_27B = "gemma3:27b")

# Mapowanie modeli na providerów
MODEL_PROVIDERS: Dict[str, ModelProvider] = {
    # OpenAI GPT-5
    Models.GPT_5: ModelProvider.OPENAI,
    Models.GPT_5_MINI: ModelProvider.OPENAI,
    Models.GPT_5_NANO: ModelProvider.OPENAI,

    # OpenAI GPT-4.1
    Models.GPT_4_1: ModelProvider.OPENAI,
    Models.GPT_4_1_MINI: ModelProvider.OPENAI,
    Models.GPT_4_1_NANO: ModelProvider.OPENAI,

    # OpenAI GPT-4o
    Models.GPT_4O: ModelProvider.OPENAI,
    Models.GPT_4O_MINI: ModelProvider.OPENAI,

    # Anthropic
    Models.CLAUDE_4_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_4_OPUS: ModelProvider.ANTHROPIC,
    Models.CLAUDE_4_1_OPUS: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_7_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_HAIKU: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_HAIKU: ModelProvider.ANTHROPIC,

    # Ollama
    Models.QWEN_CODER: ModelProvider.OLLAMA,
    Models.QWEN_CODER_32B: ModelProvider.OLLAMA,
    Models.CODESTRAL: ModelProvider.OLLAMA,

    Models.LLAMA_VISION_11B: ModelProvider.OLLAMA,
    Models.LLAMA_VISION_90B: ModelProvider.OLLAMA,
    Models.QWEN_VISION_7B: ModelProvider.OLLAMA,
    Models.GEMMA3_12B: ModelProvider.OLLAMA,
}

# Vision support
VISION_MODELS = {
    # OpenAI (multimodal)
    Models.GPT_5,
    Models.GPT_5_MINI,
    Models.GPT_5_NANO,
    Models.GPT_4_1,
    Models.GPT_4_1_MINI,
    Models.GPT_4_1_NANO,
    Models.GPT_4O,
    Models.GPT_4O_MINI,

    # Anthropic – wszystkie wymienione wspierają obraz
    Models.CLAUDE_4_SONNET,
    Models.CLAUDE_4_OPUS,
    Models.CLAUDE_4_1_OPUS,
    Models.CLAUDE_3_7_SONNET,
    Models.CLAUDE_3_5_SONNET,
    Models.CLAUDE_3_5_HAIKU,
    Models.CLAUDE_3_HAIKU,

    # Ollama vision
    Models.LLAMA_VISION_11B,
    Models.LLAMA_VISION_90B,
    Models.QWEN_VISION_7B,
    Models.GEMMA3_12B,
}

# Maksymalne limity OUTPUT (max generation / reasoning+output)
MODEL_MAX_TOKENS: Dict[str, int] = {
    # OpenAI GPT-5 (API): 128k reasoning+output
    Models.GPT_5: 128_000,
    Models.GPT_5_MINI: 128_000,
    Models.GPT_5_NANO: 128_000,

    # OpenAI GPT-4.1: 32,768
    Models.GPT_4_1: 32_768,
    Models.GPT_4_1_MINI: 32_768,
    Models.GPT_4_1_NANO: 32_768,

    # OpenAI GPT-4o: 16,384
    Models.GPT_4O: 16_384,
    Models.GPT_4O_MINI: 16_384,

    # Anthropic (oficjalna tabela)
    Models.CLAUDE_4_1_OPUS: 32_000,
    Models.CLAUDE_4_OPUS: 32_000,
    Models.CLAUDE_4_SONNET: 64_000,     # Sonnet 4
    Models.CLAUDE_3_7_SONNET: 64_000,   # z opcją 128k po headerze beta
    Models.CLAUDE_3_5_SONNET: 8_192,
    Models.CLAUDE_3_5_HAIKU: 8_192,
    Models.CLAUDE_3_HAIKU: 4_096,

    # Ollama (zależy od runtime; wartości typowe/konserwatywne)
    Models.QWEN_CODER: 4_096,
    Models.QWEN_CODER_32B: 4_096,

    # Codestral 25.01/25.08 – brak oficjalnego max output; praktycznie 4k
    Models.CODESTRAL: 4_096,

    # Vision – lokalne limity częściej runtime-owe
    Models.LLAMA_VISION_11B: 2_048,     # często spotykane
    Models.LLAMA_VISION_90B: 2_048,
    Models.QWEN_VISION_7B: 4_096,
    Models.GEMMA3_12B: 8_192,           # wg kart modeli/LM Studio
}

# INPUT context window limits (max akceptowany kontekst)
MODEL_INPUT_CONTEXT: Dict[str, int] = {
    # OpenAI GPT-5 — 272k input (400k total z 128k output)
    Models.GPT_5: 272_000,
    Models.GPT_5_MINI: 272_000,
    Models.GPT_5_NANO: 272_000,

    # OpenAI GPT-4.1 — 1,047,576
    Models.GPT_4_1: 1_047_576,
    Models.GPT_4_1_MINI: 1_047_576,
    Models.GPT_4_1_NANO: 1_047_576,

    # OpenAI GPT-4o — 128k
    Models.GPT_4O: 128_000,
    Models.GPT_4O_MINI: 128_000,

    # Anthropic — 200k (Sonnet 4 ma 1M beta z headerem)
    Models.CLAUDE_4_1_OPUS: 200_000,
    Models.CLAUDE_4_OPUS: 200_000,
    Models.CLAUDE_4_SONNET: 200_000,
    Models.CLAUDE_3_7_SONNET: 200_000,
    Models.CLAUDE_3_5_SONNET: 200_000,
    Models.CLAUDE_3_5_HAIKU: 200_000,
    Models.CLAUDE_3_HAIKU: 200_000,

    # Ollama
    Models.QWEN_CODER: 128_000,         # Qwen2.5 wspiera do 128k
    Models.QWEN_CODER_32B: 128_000,

    # Codestral 25.01/25.08 – 256k wg docs Mistrala
    Models.CODESTRAL: 256_000,

    # Vision
    Models.LLAMA_VISION_11B: 128_000,   # Llama 3.2 Vision
    Models.LLAMA_VISION_90B: 128_000,
    Models.QWEN_VISION_7B: 128_000,     # Qwen2.5VL
    Models.GEMMA3_12B: 128_000,         # Gemma 3
}

def get_model_input_limit(model_name: str) -> int:
    return MODEL_INPUT_CONTEXT.get(model_name, 32_768)

def get_model_output_limit(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 8_192)

def supports_vision(model_name: str) -> bool:
    return model_name in VISION_MODELS

def get_model_provider(model_name: str) -> ModelProvider:
    return MODEL_PROVIDERS.get(model_name, ModelProvider.OPENAI)

def resolve_model(model_identifier: str) -> str:
    """
    Przyjmuje:
      - nazwę stałej z klasy Models (np. 'GPT_4_1'), albo
      - literal nazwy modelu API (np. 'gpt-4.1').
    Zwraca literal nazwy modelu API (np. 'gpt-4.1').
    """
    if not model_identifier:
        return Models.GPT_4_1

    # wszystkie publiczne wartości z klasy Models
    items = {k: v for k, v in vars(Models).items() if not k.startswith("_")}

    # 1) literal wartości (np. 'gpt-4.1')
    if model_identifier in items.values():
        return model_identifier

    # 2) nazwa stałej (np. 'GPT_4_1')
    if model_identifier in items:
        return items[model_identifier]

    # 3) fallback: zwróć jak jest (pozwala na przyszłe modele bez edycji kodu)
    return model_identifier