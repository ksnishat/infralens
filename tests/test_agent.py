"""Tests for the safety-report agent's model resolution.

The bug these cover: the agent asked Ollama for a tag that was not pulled and
had no fallback, so every report silently degraded to the template while the
README claimed Llama 3. The API stayed healthy and the response looked
plausible, which is why it survived.
"""
from __future__ import annotations

from src.core.agent import (
    DEFAULT_LLM_MODEL,
    _template_based_report,
    resolve_llm_model,
)


class TestResolveLlmModel:
    def test_uses_the_requested_tag_when_it_is_pulled(self):
        chosen, warning = resolve_llm_model("llama3.2:1b", "http://x", ["llama3.2:1b"])
        assert chosen == "llama3.2:1b"
        assert warning is None

    def test_falls_back_to_a_same_family_tag(self):
        chosen, warning = resolve_llm_model("llama3.2", "http://x", ["llama3.2:1b", "llava:7b"])
        assert chosen == "llama3.2:1b"
        assert warning is not None
        assert "llama3.2" in warning

    def test_falls_back_to_any_tag_when_the_family_is_absent(self):
        chosen, warning = resolve_llm_model("llama3", "http://x", ["mistral:7b"])
        assert chosen == "mistral:7b"
        assert warning is not None

    def test_keeps_the_requested_tag_when_the_server_is_unreachable(self):
        # An unreachable server must surface as an error from the LLM call, not
        # be masked by a substitution.
        chosen, warning = resolve_llm_model("llama3.2", "http://x", [])
        assert chosen == "llama3.2"
        assert warning is None

    def test_defaults_when_nothing_is_requested(self):
        chosen, _ = resolve_llm_model(None, "http://x", [DEFAULT_LLM_MODEL])
        assert chosen == DEFAULT_LLM_MODEL


class TestTemplateFallback:
    def test_template_is_marked_as_such(self):
        # Without the marker a template report is indistinguishable from an LLM
        # report, which is how a silently degraded agent looks like a working one.
        report = _template_based_report("Found rust with 0.85 confidence.")
        assert "template fallback" in report

    def test_template_counts_rust_lines(self):
        report = _template_based_report("rust a\nrust b\npitting c")
        assert "Total rust spots detected: 2" in report

    def test_template_handles_empty_input(self):
        report = _template_based_report("")
        assert "Total rust spots detected: 0" in report


class TestOllamaHostDefault:
    def test_default_host_is_localhost(self, monkeypatch):
        # host.docker.internal resolves inside a container but not from a
        # host-run process, and the API runs on the host.
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        from src.core.agent import _get_ollama_url

        assert _get_ollama_url() == "http://localhost:11434"

    def test_env_override_is_respected(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
        from src.core.agent import _get_ollama_url

        assert _get_ollama_url() == "http://ollama:11434"
