from dataclasses import dataclass
import os


# ---- Core domain defaults ----
# Granularity fixed to annual per current design
GRANULARITY = "year"

# Scope limitation: S1 + S2 only for now
SCOPES = ("S1", "S2")

# Transfer rules
ALLOW_NEGATIVE_FLOWS = False
DEFAULT_ALLOCATION_METHOD = "average"  # average split across multiple producers


# ---- LLM configuration ----
# Built-in defaults so用户无需环境变量即可直接使用 SiliconFlow DeepSeek-V3

LLM_DEFAULT_TEMPERATURE = 0.2
LLM_DEFAULT_TIMEOUT = 60
@dataclass
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    timeout: int


def load_llm_config() -> LLMConfig:
    """
    Provider-agnostic LLM config with SiliconFlow (DeepSeek) as sensible default.

    Environment variables (priority order):
    - LLM_PROVIDER: 'siliconflow' | 'openai' | 'azure' | 'auto'
    - LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_TEMPERATURE / LLM_TIMEOUT
    - Backward compatibility fallbacks for SiliconFlow/OpenAI:
        SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY, SILICONFLOW_MODEL
        OPENAI_API_KEY, OPENAI_MODEL
    """

    # Detect provider if not explicitly set
    provider = os.environ.get("LLM_PROVIDER") or LLM_DEFAULT_PROVIDER
    if not provider:
        if os.environ.get("SILICONFLOW_API_KEY"):
            provider = "siliconflow"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = LLM_DEFAULT_PROVIDER  # default to SiliconFlow

    # Base URL
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("SILICONFLOW_BASE_URL")
        or ("https://api.openai.com/v1" if provider == "openai" else LLM_DEFAULT_BASE_URL)
    )

    # API key precedence: LLM_*, then provider-specific
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or LLM_DEFAULT_API_KEY
    )

    # Model precedence: LLM_*, then provider-specific, then sensible default
    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("SILICONFLOW_MODEL")
        or (os.environ.get("OPENAI_MODEL") if provider == "openai" else None)
        or LLM_DEFAULT_MODEL
    )

    # Temperature and timeout
    temperature = float(os.environ.get("LLM_TEMPERATURE") or os.environ.get("SILICONFLOW_TEMPERATURE") or LLM_DEFAULT_TEMPERATURE)
    timeout = int(os.environ.get("LLM_TIMEOUT") or LLM_DEFAULT_TIMEOUT)

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )



# ---- MySQL configuration ----
@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = "QWEmnb12345"
    database: str = "carbon_footprint"
    charset: str = "utf8mb4"





# 直接在此处修改为你的实际 MySQL 连接信息（Navicat 同样配置参数）
MYSQL = MySQLConfig()


