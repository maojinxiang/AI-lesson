"""Minimal LLM client wrapper for DeepSeek only.
"""

from __future__ import annotations

from typing import Dict, List


# 只使用 DeepSeek：把你的真实 API Key 填在这里。
DEEPSEEK_API_KEY = "sk-be2805aa97d14608a643b80b4f790a1b"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


class LLMClient:
    """Thin wrapper around DeepSeek Chat Completions API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEEPSEEK_MODEL
        self.base_url = DEEPSEEK_BASE_URL
        self.api_key = DEEPSEEK_API_KEY

    def _validated_api_key(self) -> str:
        key = (self.api_key or "").strip()
        if not key:
            raise RuntimeError("缺少 API Key。请在 llm_client.py 的 DEEPSEEK_API_KEY 中填写。")

        placeholder_hints = ("你的", "your", "api_key", "key_here", "示例", "sk-这里")
        lower_key = key.lower()
        if any(hint in lower_key for hint in placeholder_hints):
            raise RuntimeError("检测到占位符 Key，请把 DEEPSEEK_API_KEY 改成真实密钥。")

        try:
            key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError("API Key 含有非 ASCII 字符，请检查是否混入中文或空格。") from exc

        return key

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        api_key = self._validated_api_key()

        try:
            from openai import OpenAI
            from openai import APIConnectionError
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "未安装 openai 包。请先执行: pip install openai"
            ) from exc

        # 关闭系统代理继承，避免被错误的 HTTP(S)_PROXY 导致 TLS 握手失败。
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        client = OpenAI(api_key=api_key, base_url=self.base_url, http_client=http_client)

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except APIConnectionError as exc:
            raise RuntimeError(
                "连接 DeepSeek 失败（TLS/网络问题）。请检查：\n"
                "1) 本机网络可访问 https://api.deepseek.com\n"
                "2) 是否开启了代理/VPN；如开启请确认代理可用\n"
                "3) 防火墙或校园网是否拦截 HTTPS 出站连接"
            ) from exc
        return (resp.choices[0].message.content or "").strip()
