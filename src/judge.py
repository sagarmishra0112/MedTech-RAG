"""
src/judge.py — Shared Judge LLM Factory
=========================================
Provides a single `build_judge_llm()` function used by:
  - src/evaluate.py  (offline RAGAS evaluation)
  - src/api.py       (runtime self-correction in the agent loop)

Configure which provider acts as judge via .env:
    JUDGE_LLM=anthropic   → Claude 3.5 Haiku  (default, recommended)
    JUDGE_LLM=google      → Gemini 1.5 Flash
    JUDGE_LLM=openai      → GPT-4o-mini  ⚠️ avoid — self-grading bias

The judge is intentionally a different provider from the generator (gpt-4o-mini)
so grading is independent and unbiased.
"""

import os
import importlib

from ragas.llms import LangchainLLMWrapper


# Provider registry: JUDGE_LLM value → constructor config.
# model_kwarg differs per provider: ChatAnthropic uses 'model_name', others use 'model'.
_JUDGE_PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "import": "langchain_anthropic.ChatAnthropic",
        "model_kwarg": "model_name",
        "model": "claude-haiku-4-5-20251001",
        "key_var": "ANTHROPIC_API_KEY",
        "label": "Claude Haiku 4.5 (Anthropic)",
    },
    "google": {
        "import": "langchain_google_genai.ChatGoogleGenerativeAI",
        "model_kwarg": "model",
        "model": "gemini-1.5-flash",
        "key_var": "GOOGLE_API_KEY",
        "label": "Gemini 1.5 Flash (Google)",
    },
    "openai": {
        "import": "langchain_openai.ChatOpenAI",
        "model_kwarg": "model",
        "model": "gpt-4o-mini",
        "key_var": "OPENAI_API_KEY",
        "label": "GPT-4o-mini (OpenAI) ⚠️  SELF-GRADING — same family as generator",
    },
}


def build_judge_llm() -> LangchainLLMWrapper:
    """
    Builds and returns a RAGAS-compatible LangchainLLMWrapper for the judge role.

    Reads JUDGE_LLM from the environment (default: 'anthropic').
    Fails fast with a clear EnvironmentError if the required API key is missing.

    Returns:
        LangchainLLMWrapper wrapping the configured LangChain chat model.

    Raises:
        ValueError: If JUDGE_LLM is set to an unrecognised provider name.
        EnvironmentError: If the required API key env var is missing.
        ImportError: If the provider's LangChain package is not installed.
    """
    provider = os.getenv("JUDGE_LLM", "anthropic").lower()

    if provider not in _JUDGE_PROVIDERS:
        raise ValueError(
            f"Unknown JUDGE_LLM='{provider}'. "
            f"Choose one of: {list(_JUDGE_PROVIDERS.keys())}"
        )

    config = _JUDGE_PROVIDERS[provider]
    api_key = os.getenv(config["key_var"])

    if not api_key:
        raise EnvironmentError(
            f"JUDGE_LLM='{provider}' but {config['key_var']} is not set in .env.\n"
            f"Add {config['key_var']} to .env, or change JUDGE_LLM to a provider whose key you have."
        )

    if provider == "openai":
        print(
            "[WARN] JUDGE_LLM=openai - same model family as the generator (self-grading bias).\n"
            "   Set JUDGE_LLM=anthropic for unbiased cross-provider evaluation."
        )

    print(f"[Judge] LLM: {config['label']}")

    # Dynamically import only the required provider package
    module_path, class_name = config["import"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        LLMClass = getattr(module, class_name)
    except ImportError:
        pkg = module_path.replace("_", "-")
        raise ImportError(
            f"Provider package not installed. Run: pip install {pkg}"
        )

    langchain_llm = LLMClass(**{config["model_kwarg"]: config["model"], "temperature": 0})
    return LangchainLLMWrapper(langchain_llm)


def build_judge_llm_raw():
    """
    Same as build_judge_llm() but returns the raw LangChain chat model
    (not wrapped in LangchainLLMWrapper).

    Use this for runtime inference (agent grading) where you call llm.invoke()
    directly. Use build_judge_llm() (wrapped) for RAGAS evaluate() calls.
    """
    provider = os.getenv("JUDGE_LLM", "anthropic").lower()

    if provider not in _JUDGE_PROVIDERS:
        raise ValueError(
            f"Unknown JUDGE_LLM='{provider}'. "
            f"Choose one of: {list(_JUDGE_PROVIDERS.keys())}"
        )

    config = _JUDGE_PROVIDERS[provider]
    api_key = os.getenv(config["key_var"])

    if not api_key:
        raise EnvironmentError(
            f"JUDGE_LLM='{provider}' but {config['key_var']} is not set in .env.\n"
            f"Add {config['key_var']} to .env, or change JUDGE_LLM to a provider whose key you have."
        )

    module_path, class_name = config["import"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        LLMClass = getattr(module, class_name)
    except ImportError:
        pkg = module_path.replace("_", "-")
        raise ImportError(f"Run: pip install {pkg}")

    return LLMClass(**{config["model_kwarg"]: config["model"], "temperature": 0})
