"""Ollama local API client for Offline Study AI.

Communicates directly with the local Ollama daemon on http://127.0.0.1:11434
to generate answers using the local qwen2.5:1.5b model.
"""

import os
from typing import Any, Optional
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_TIMEOUT = 120.0


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the specified model is not available in Ollama."""
    pass


def check_ollama_availability(
    base_url: str = OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    timeout: float = 3.0
) -> None:
    """Checks whether the Ollama service is reachable and the model is loaded.

    Raises:
        OllamaConnectionError: If Ollama is offline or unreachable.
        OllamaModelNotFoundError: If the model is not found in Ollama.
    """
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=timeout)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
        raise OllamaConnectionError(
            f"Ollama is not running or unreachable at {base_url}. Please ensure Ollama is running locally."
        ) from e
    except Exception as e:
        raise OllamaConnectionError(
            f"Failed to connect to Ollama at {base_url}: {str(e)}"
        ) from e

    if response.status_code != 200:
        raise OllamaConnectionError(
            f"Ollama health check returned status code {response.status_code}."
        )

    try:
        data = response.json()
    except Exception as e:
        raise OllamaError(f"Failed to parse Ollama tags response: {str(e)}") from e

    available_models = [m.get("name", "") for m in data.get("models", [])]

    # Handle model tag matching (e.g., 'qwen2.5:1.5b' vs 'qwen2.5:1.5b-latest' or full tag)
    model_matched = any(
        m == model
        or m == f"{model}:latest"
        or (":" not in model and m.startswith(f"{model}:"))
        for m in available_models
    )

    if not model_matched:
        raise OllamaModelNotFoundError(
            f"Model '{model}' is not installed in Ollama. Available models: {available_models}. "
            f"Please run 'ollama pull {model}' to download it."
        )


def generate_rag_answer(
    prompt: str,
    system_prompt: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.1,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
) -> str:
    """Calls Ollama /api/generate endpoint to generate an answer.

    Args:
        prompt: User question with context.
        system_prompt: System instructions constraining the answer.
        base_url: Base URL of local Ollama server.
        model: Model name (default: qwen2.5:1.5b).
        temperature: Sampling temperature (low for factual adherence).
        timeout: Request timeout in seconds.
        max_tokens: Optional maximum tokens to predict (num_predict).

    Returns:
        The generated answer text.

    Raises:
        OllamaConnectionError: If Ollama is unreachable.
        OllamaModelNotFoundError: If the model is not found.
        OllamaError: For other generation errors.
    """
    options: dict[str, Any] = {
        "temperature": temperature,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = httpx.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
        raise OllamaConnectionError(
            f"Failed to connect to Ollama at {base_url}. Ensure Ollama is running."
        ) from e
    except httpx.TimeoutException as e:
        raise OllamaError("Ollama generation request timed out.") from e
    except Exception as e:
        raise OllamaError(f"Unexpected error communicating with Ollama: {str(e)}") from e

    if response.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{model}' was not found in Ollama (HTTP 404)."
        )
    if response.status_code != 200:
        raise OllamaError(
            f"Ollama returned HTTP {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        raise OllamaError(f"Failed to parse JSON response from Ollama: {str(e)}") from e
