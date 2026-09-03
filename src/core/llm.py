"""OpenAI 兼容接口的极简客户端。

刻意不引入任何 LLM 封装框架：本项目只需要"发一次 chat 请求、拿回一段文本"，
用 httpx 直接打 HTTP 即可，依赖越少越不容易在演示现场出问题。
密钥只从环境变量读取，配置文件里只写环境变量名。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .config import Settings, load_settings


class LLMUnavailable(RuntimeError):
    """模型不可用（未配置密钥、网络失败、重试耗尽）。调用方据此降级。"""


@dataclass
class ChatClient:
    endpoint: str
    model: str
    api_key: Optional[str]
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    enabled: bool = True
    trust_env_proxy: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None, section: str = "llm") -> "ChatClient":
        settings = settings or load_settings()
        return cls(
            endpoint=settings.get(f"{section}.endpoint", ""),
            model=settings.get(f"{section}.model", ""),
            api_key=settings.secret(f"{section}.api_key_env"),
            temperature=float(settings.get(f"{section}.temperature", 0.0)),
            max_tokens=int(settings.get(f"{section}.max_tokens", 2048)),
            timeout_seconds=float(settings.get(f"{section}.timeout_seconds", 120)),
            max_retries=int(settings.get(f"{section}.max_retries", 3)),
            retry_backoff_seconds=float(settings.get(f"{section}.retry_backoff_seconds", 2.0)),
            enabled=bool(settings.get(f"{section}.enabled", True)),
            trust_env_proxy=bool(settings.get(f"{section}.trust_env_proxy", False)),
        )

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.endpoint and self.model and self.api_key)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.ready:
            raise LLMUnavailable("未配置可用的模型端点或密钥，已切换到降级通道")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(trust_env=self.trust_env_proxy) as client:
                    response = client.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        raise LLMUnavailable(f"模型调用重试 {self.max_retries} 次仍失败: {last_error}")


def parse_json_block(text: str) -> Any:
    """模型常把 JSON 裹在围栏或前后寒暄里，这里做一次宽容解析。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("模型返回中找不到可解析的 JSON")
