# llm_client.py
# Jednolity klient czatu LLM z podklientami dla providerów: OpenAI, Anthropic, Ollama.
# Wymaga: models.py (z wczytywaniem models.yaml), requests (dla Ollama).

from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional, Tuple

# -- konfiguracja modeli w YAML:
import models  # Twój plik z wcześniejszego kroku (ładuje models.yaml)

# --- OpenAI (opcjonalnie) ---
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# --- Anthropic (opcjonalnie) ---
try:
    import anthropic as _anthropic
except Exception:
    _anthropic = None

# --- Ollama przez HTTP ---
import requests


Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class BaseProviderClient:
    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        raise NotImplementedError


# ===================== OpenAI =====================
class OpenAIClient(BaseProviderClient):
    def __init__(self):
        if _OpenAI is None:
            raise RuntimeError("Brak pakietu openai. Zainstaluj: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("Brak OPENAI_API_KEY / OPENAI_KEY w środowisku.")
        self.client = _OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": float(temperature),
            "messages": messages,
        }
        if max_output_tokens is not None:
            kwargs["max_tokens"] = int(max_output_tokens)
        if seed is not None:
            kwargs["seed"] = int(seed)

        resp = self.client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


# ===================== Anthropic =====================
class AnthropicClient(BaseProviderClient):
    def __init__(self):
        if _anthropic is None:
            raise RuntimeError("Brak pakietu anthropic. Zainstaluj: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")
        if not api_key:
            raise RuntimeError("Brak ANTHROPIC_API_KEY / ANTHROPIC_KEY w środowisku.")
        self.client = _anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _split_system_and_msgs(messages: List[Message]) -> Tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        chat_parts: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                chat_parts.append({"role": role, "content": content})
            else:
                # ignoruj inne role w trybie Anthropic
                pass
        system_text = "\n".join(system_parts).strip() if system_parts else ""
        return system_text, chat_parts

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        system_text, chat_parts = self._split_system_and_msgs(messages)

        # Anthropic wymaga max_tokens i messages jako listy {"role","content"}
        max_tokens = int(max_output_tokens) if max_output_tokens is not None else max(
            1024, models.get_model_output_limit(model)
        )

        # seed jest w Anthropic eksperymentalny — pomijamy jeśli brak wsparcia
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "messages": chat_parts,
        }
        if system_text:
            kwargs["system"] = system_text
        # brak gwarantowanego wsparcia seed → nie ustawiamy

        resp = self.client.messages.create(**kwargs)
        # łączymy treść z bloków (Anthropic może zwracać kilka content bloków)
        out_chunks: List[str] = []
        for blk in resp.content or []:
            if getattr(blk, "type", "") == "text":
                out_chunks.append(getattr(blk, "text", "") or "")
            elif isinstance(blk, dict) and blk.get("type") == "text":
                out_chunks.append(blk.get("text") or "")
        return "\n".join([x for x in out_chunks if x]).strip()


# ===================== Ollama =====================
class OllamaClient(BaseProviderClient):
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        # brak klucza – lokalny runtime

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        # API /api/chat: https://github.com/ollama/ollama/blob/main/docs/api.md#chat
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": float(temperature),
            },
            "stream": False,
        }
        # mapowanie max_output_tokens → num_predict
        if max_output_tokens is not None:
            payload["options"]["num_predict"] = int(max_output_tokens)
        if seed is not None:
            payload["options"]["seed"] = int(seed)

        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        # odpowiedź zwykle w data["message"]["content"]
        msg = (data.get("message", {}) or {}).get("content", "")
        if msg:
            return msg.strip()

        # alternatywnie pełny zwrot
        return json.dumps(data, ensure_ascii=False)


# ===================== Fabryka i klient zewnętrzny =====================
class LLMClient:
    """
    Jednolity interfejs:
      chat(model, messages, temperature, max_output_tokens=None, seed=None) -> str
    Provider wybierany na podstawie models.yaml (provider dla danego modelu).
    """

    def __init__(self):
        self._openai: Optional[OpenAIClient] = None
        self._anthropic: Optional[AnthropicClient] = None
        self._ollama: Optional[OllamaClient] = None

    def _get_provider(self, model: str) -> str:
        return models.get_model_provider(model).value  # "openai" | "anthropic" | "ollama"

    def _ensure_client(self, provider: str) -> BaseProviderClient:
        if provider == "openai":
            if self._openai is None:
                self._openai = OpenAIClient()
            return self._openai
        elif provider == "anthropic":
            if self._anthropic is None:
                self._anthropic = AnthropicClient()
            return self._anthropic
        elif provider == "ollama":
            if self._ollama is None:
                self._ollama = OllamaClient()
            return self._ollama
        else:
            raise RuntimeError(f"Nieznany provider: {provider}")

    def chat(
        self,
        model: str,
        messages: List[Message],
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        provider = self._get_provider(model)
        client = self._ensure_client(provider)

        # Fallback na limit z models.yaml, jeśli nie podano
        if max_output_tokens is None:
            max_output_tokens = models.get_model_output_limit(model)

        return client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
