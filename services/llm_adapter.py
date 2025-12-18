from typing import Optional, List, Dict, Any
from config import load_llm_config

try:
    import requests  # type: ignore
except Exception:
    requests = None


def _require_requests():
    if requests is None:
        raise RuntimeError("缺少 requests 库，请先安装：pip install requests")


def _sf_request(payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    _require_requests()
    cfg = load_llm_config()
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code == 401:
        raise RuntimeError("AI 鉴权失败(401)。请检查 API Key、模型权限与 base_url。")
    if not resp.ok:
        raise RuntimeError(f"AI 调用失败：HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def _merge_params(cfg, temperature: Optional[float], top_p: Optional[float], top_k: Optional[int], max_tokens: Optional[int]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "temperature": cfg.temperature if temperature is None else float(temperature),
    }
    if top_p is not None:
        params["top_p"] = float(top_p)
    if top_k is not None:
        params["top_k"] = int(top_k)
    if max_tokens is not None:
        params["max_tokens"] = int(max_tokens)
    return params


def generate_with_llm(
    prompt: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """单轮对话：硅基流动 DeepSeek 模型，支持可调超参数。"""
    cfg = load_llm_config()
    if not cfg.api_key:
        raise RuntimeError("AI 服务未配置。")
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(_merge_params(cfg, temperature, top_p, top_k, max_tokens))
    data = _sf_request(payload, timeout=cfg.timeout)
    return data["choices"][0]["message"]["content"].strip()


def chat_with_llm(
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """多轮对话：messages 为 [{'role': 'user'|'assistant'|'system', 'content': '...'}]"""
    cfg = load_llm_config()
    if not cfg.api_key:
        raise RuntimeError("AI 服务未配置。")
    payload = {
        "model": cfg.model,
        "messages": messages,
    }
    payload.update(_merge_params(cfg, temperature, top_p, top_k, max_tokens))
    data = _sf_request(payload, timeout=cfg.timeout)
    return data["choices"][0]["message"]["content"].strip()
