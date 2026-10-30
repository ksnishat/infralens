"""
InfraLens AI Safety Report Agent.

This module provides the Llama 3 agent for generating ISO-compliant
safety reports from Rust detection data.

Usage:
    from src.core.agent import generate_safety_report

    report = generate_safety_report("Found rust with 0.85 confidence.")
"""

import logging
import os

logger = logging.getLogger(__name__)


def _get_ollama_url() -> str:
    """Get the Ollama API URL.

    Defaults to localhost, matching the other projects. The previous default was
    ``host.docker.internal``, which resolves inside a container but not from a
    host-run process — and the API runs on the host. Every report therefore fell
    back to the template while the README claimed Llama 3.
    """
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


DEFAULT_LLM_MODEL = "llama3.2"


def _list_ollama_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Return the model tags the Ollama server has pulled.

    Returns an empty list when the server is unreachable, so callers can treat
    "cannot tell" the same as "nothing available" without special-casing.
    """
    try:
        import requests

        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return []
        return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def resolve_llm_model(
    requested: str | None, base_url: str, available: list[str] | None = None
) -> tuple[str, str | None]:
    """Pick an Ollama model tag that actually exists.

    A configured tag that is not pulled is the worst case: the API starts,
    ``/health`` reports the agent as available, and every report silently
    degrades to the template while the README claims Llama 3. This resolves to a
    real tag and reports what it did.

    Returns:
        ``(model_name, warning)``. ``warning`` is None when the requested tag was
        used as-is.
    """
    tags = available if available is not None else _list_ollama_models(base_url)
    wanted = requested or DEFAULT_LLM_MODEL

    if not tags:
        # Server unreachable: keep the requested tag so the error surfaces from
        # the LLM call rather than being masked by a substitution.
        return wanted, None

    if wanted in tags:
        return wanted, None

    # Prefer a same-family tag (llama3.2 -> llama3.2:1b) before falling back to
    # an arbitrary model, so a quantised variant of the intended model wins over
    # an unrelated one.
    family = wanted.split(":")[0]
    same_family = [t for t in tags if t.split(":")[0] == family]
    chosen = same_family[0] if same_family else tags[0]

    warning = (
        f"OLLAMA_MODEL={wanted!r} is not pulled (available: {', '.join(tags)}). "
        f"Using {chosen!r} instead. Set OLLAMA_MODEL to silence this."
    )
    return chosen, warning


def _get_ollama_model() -> str:
    """Get the Ollama model tag to use, resolving it against what is pulled."""
    requested = os.getenv("OLLAMA_MODEL")
    chosen, warning = resolve_llm_model(requested, _get_ollama_url())
    if warning:
        logger.warning(warning)
    return chosen


def generate_safety_report(detection_data: str) -> str:
    """
    Generate a technical safety report from detection data.

    Args:
        detection_data: Structured detection results from YOLOv8 inference.
            Format: "Found rust with 0.85 confidence.\nFound pitting with 0.62 confidence."

    Returns:
        A structured safety report in German/English.
    """
    # If Ollama is not available, return a template-based report
    if not _is_ollama_available():
        return _template_based_report(detection_data)

    # Use Llama 3 for report generation
    try:
        import requests

        url = f"{_get_ollama_url()}/api/generate"
        payload = {
            "model": _get_ollama_model(),
            "prompt": _build_safety_report_prompt(detection_data),
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "")
            if text:
                return text
            logger.warning("Ollama returned an empty response")
            return _template_based_report(detection_data)
        logger.warning(f"Ollama returned status {response.status_code}")
        return _template_based_report(detection_data)
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return _template_based_report(detection_data)


def _is_ollama_available() -> bool:
    """Check if Ollama is reachable."""
    try:
        import requests
        url = f"{_get_ollama_url()}/api/tags"
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _template_based_report(detection_data: str) -> str:
    """Generate a report using template-based approach (no LLM)."""
    lines = detection_data.strip().split("\n") if detection_data else []
    rust_count = len([line for line in lines if "rust" in line.lower()])

    # Marked as a template so a reader can tell it apart from an LLM report.
    # Without this the two are indistinguishable, which is how a silently
    # degraded agent looks like a working one.
    report = "=== ISO-STYLE SAFETY REPORT (template fallback) ===\n"
    report += f"Total rust spots detected: {rust_count}\n"
    report += "Severity: MEDIUM\n"
    report += "Recommended action: Schedule inspection within 7 days\n"
    report += "Compliance: ISO 13401 (Safety of machinery)\n"
    report += "=============================\n"
    return report


def _build_safety_report_prompt(detection_data: str) -> str:
    """Build the prompt for Llama 3 safety report generation."""
    return f"""You are an industrial safety inspector for a German manufacturing facility.
Generate a structured safety report from the following detection data.

Detection Data:
{detection_data}

Format the report as:
1. Summary of findings
2. Severity assessment (LOW / MEDIUM / HIGH / CRITICAL)
3. Root cause analysis (brief)
4. Recommended action (with timeline)
5. Compliance reference (ISO 13401 / DGUV)

Write in German and English. Be precise and factual."""


# Expose for FastAPI app
__all__ = ["generate_safety_report"]
