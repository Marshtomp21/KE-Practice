"""Small, provider-neutral HTTP clients for the API-adapted KG²RAG run.

No credential is stored in this repository.  The clients deliberately use the
documented HTTP endpoints instead of modifying KG2RAG's Ollama-only upstream
implementation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import httpx


class APIError(RuntimeError):
    """A remote provider returned an unusable response."""


def load_env_file(path: Optional[Path]) -> Dict[str, str]:
    """Read a minimal KEY=VALUE file without exporting secrets to the shell."""
    values = dict(os.environ)
    if path is None or not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def require(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise APIError(
            f"Missing {name}. Copy config/api.env.example to config/api.env and fill it locally."
        )
    return value


@dataclass
class APISettings:
    llm_api_key: str
    llm_endpoint: str
    llm_model: str
    embed_api_key: str
    embed_endpoint: str
    embed_model: str
    rerank_api_key: str
    rerank_endpoint: str
    rerank_model: str

    @classmethod
    def from_values(cls, values: Mapping[str, str], require_all: bool = True) -> "APISettings":
        get = lambda key, default="": values.get(key, default).strip()
        if require_all:
            return cls(
                require(values, "REPRO_LLM_API_KEY"),
                get("REPRO_LLM_ENDPOINT", "https://api.deepseek.com/chat/completions"),
                get("REPRO_LLM_MODEL", "deepseek-v4-flash"),
                require(values, "REPRO_EMBED_API_KEY"),
                get("REPRO_EMBED_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings"),
                get("REPRO_EMBED_MODEL", "BAAI/bge-m3"),
                require(values, "REPRO_RERANK_API_KEY"),
                get("REPRO_RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
                get("REPRO_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            )
        return cls(
            get("REPRO_LLM_API_KEY"), get("REPRO_LLM_ENDPOINT", "https://api.deepseek.com/chat/completions"),
            get("REPRO_LLM_MODEL", "deepseek-v4-flash"), get("REPRO_EMBED_API_KEY"),
            get("REPRO_EMBED_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings"),
            get("REPRO_EMBED_MODEL", "BAAI/bge-m3"), get("REPRO_RERANK_API_KEY"),
            get("REPRO_RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
            get("REPRO_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        )


def _post(endpoint: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=90.0, trust_env=False) as client:
                response = client.post(endpoint, headers=headers, json=payload)
            if response.status_code >= 400:
                raise APIError(f"{response.status_code} from {endpoint}: {response.text[:500]}")
            data = response.json()
            if not isinstance(data, dict):
                raise APIError(f"Unexpected non-object response from {endpoint}")
            return data
        except (httpx.HTTPError, ValueError, APIError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.5 * (attempt + 1))
    raise APIError(str(last_error))


class ChatClient:
    def __init__(self, settings: APISettings):
        self.settings = settings

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 800) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        # DeepSeek's thinking models can spend the whole token budget in
        # reasoning_content and leave content empty.  Extraction and QA require
        # a normal answer body, matching the adapters used by the other runs.
        if "deepseek" in self.settings.llm_model.casefold() or "deepseek" in self.settings.llm_endpoint.casefold():
            payload["thinking"] = {"type": "disabled"}
        data = _post(self.settings.llm_endpoint, self.settings.llm_api_key, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"Unexpected chat response: {data}") from exc
        if not isinstance(content, str):
            raise APIError("Chat response content is not text")
        return content


class EmbeddingClient:
    def __init__(self, settings: APISettings):
        self.settings = settings

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        items = list(texts)
        data = _post(self.settings.embed_endpoint, self.settings.embed_api_key, {
            "model": self.settings.embed_model, "input": items, "encoding_format": "float",
        })
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise APIError(f"Unexpected embedding response: {data}") from exc
        if len(vectors) != len(items):
            raise APIError(f"Embedding count mismatch: sent {len(items)}, received {len(vectors)}")
        return vectors


class RerankClient:
    def __init__(self, settings: APISettings):
        self.settings = settings

    def rerank(self, query: str, documents: List[str], top_n: int) -> List[Dict[str, Any]]:
        data = _post(self.settings.rerank_endpoint, self.settings.rerank_api_key, {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        })
        rows = data.get("results")
        if not isinstance(rows, list):
            raise APIError(f"Unexpected rerank response: {data}")
        return rows
