"""Tests for the bounded tool-calling loop.

The loop is the part of the agent that can be wrong in ways a reader cannot
see: it can loop forever, swallow a malformed tool call, or return a partial
answer as if it were complete. Every branch is tested here with a fake LLM, so
none of these tests need a model.
"""
from __future__ import annotations

import json

import pytest

from src.core.agent_loop import (
    DEFAULT_MAX_STEPS,
    AgentRun,
    Step,
    check_iso_thresholds,
    fetch_structure_history,
    parse_tool_call,
    run_agent,
)


class FakeLLM:
    """Returns scripted responses in order, then repeats the last one."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        idx = min(len(self.prompts) - 1, len(self.responses) - 1)
        return self.responses[idx]


def tool_call(name: str, **args) -> str:
    return json.dumps({"tool": name, "arguments": args})


class TestParseToolCall:
    def test_returns_none_when_there_is_no_call(self):
        """No tool call is how a model says it is ready to answer."""
        assert parse_tool_call("Here is the final report. Severity: HIGH.") is None

    def test_extracts_a_call_embedded_in_prose(self):
        text = 'I need data. {"tool": "check_iso_thresholds", "arguments": {"corrosion_pct": 12}}'
        call = parse_tool_call(text)
        assert call is not None
        assert call["tool"] == "check_iso_thresholds"
        assert call["arguments"]["corrosion_pct"] == 12

    def test_malformed_json_raises_so_it_can_be_counted(self):
        """A parse failure must be distinguishable from 'no call'."""
        with pytest.raises(ValueError, match="malformed"):
            parse_tool_call('{"tool": "check_iso_thresholds", "arguments": {')

    def test_missing_tool_key_raises(self):
        with pytest.raises(ValueError, match="missing 'tool'"):
            parse_tool_call('{"arguments": {"x": 1}}')


class TestIsoThresholds:
    @pytest.mark.parametrize(
        "pct,expected",
        [(0.0, "C1"), (5.0, "C2"), (25.0, "C3"), (50.0, "C4"), (100.0, "C5")],
    )
    def test_maps_percentage_to_category(self, pct, expected):
        r = check_iso_thresholds(pct)
        assert r.ok
        assert expected in r.content

    def test_rejects_non_numeric(self):
        r = check_iso_thresholds("lots")
        assert not r.ok
        assert "must be a number" in r.error

    def test_rejects_out_of_range(self):
        assert not check_iso_thresholds(-1).ok
        assert not check_iso_thresholds(101).ok

    def test_states_that_the_mapping_is_an_approximation(self):
        """ISO classifies by mass loss; the detector reports area fraction.

        Presenting one as the other would be a quiet over-claim, which is the
        failure mode this project has hit before.
        """
        r = check_iso_thresholds(25.0)
        assert "Approximation" in r.content


class TestStructureHistory:
    def test_reports_first_inspection_when_empty(self):
        r = fetch_structure_history("POLE-1", store={})
        assert r.ok
        assert "first inspection" in r.content

    def test_returns_records_when_present(self):
        store = {
            "POLE-1": [
                {"date": "2026-01-01", "finding": "light rust", "severity": "LOW"},
                {"date": "2026-06-01", "finding": "pitting", "severity": "HIGH"},
            ]
        }
        r = fetch_structure_history("POLE-1", store=store)
        assert r.ok
        assert "light rust" in r.content
        assert "pitting" in r.content


class TestLoopHappyPath:
    def test_returns_the_answer_when_no_tool_is_requested(self):
        llm = FakeLLM(["Final report: severity HIGH."])
        run = run_agent("rust 0.9", llm)
        assert run.answer == "Final report: severity HIGH."
        assert run.tool_calls == 0
        assert not run.budget_exhausted

    def test_calls_a_tool_then_answers(self):
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                "Final report: severity MEDIUM.",
            ]
        )
        run = run_agent("rust 0.9", llm)
        assert run.tool_calls == 1
        assert run.steps[0].tool == "check_iso_thresholds"
        assert run.steps[0].ok
        assert "C3" in run.steps[0].result
        assert run.answer == "Final report: severity MEDIUM."

    def test_tool_result_is_fed_back_into_the_next_prompt(self):
        """The loop is only useful if the model can see what the tool returned."""
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                "done",
            ]
        )
        run_agent("rust 0.9", llm)
        assert "C3" in llm.prompts[1]


class TestLoopBudget:
    def test_stops_at_max_steps(self):
        llm = FakeLLM([tool_call("check_iso_thresholds", corrosion_pct=10)])
        run = run_agent("rust", llm, max_steps=3)
        assert run.budget_exhausted
        assert run.tool_calls == 3

    def test_budget_exhaustion_is_recorded_in_the_audit_trail(self):
        """A partial answer must be distinguishable from a complete one."""
        llm = FakeLLM([tool_call("check_iso_thresholds", corrosion_pct=10)])
        run = run_agent("rust", llm, max_steps=2)
        trail = run.audit_trail()
        assert trail["budget_exhausted"] is True
        assert trail["steps"] == 2

    def test_asks_for_a_final_answer_after_exhaustion(self):
        llm = FakeLLM([tool_call("check_iso_thresholds", corrosion_pct=10)])
        run = run_agent("rust", llm, max_steps=2)
        assert "used all available tool calls" in llm.prompts[-1]
        assert run.answer

    def test_default_budget_is_four(self):
        assert DEFAULT_MAX_STEPS == 4


class TestLoopFailureModes:
    def test_malformed_json_is_counted_and_the_loop_continues(self):
        llm = FakeLLM(
            [
                '{"tool": "check_iso_thresholds", "arguments": {',
                "Final report.",
            ]
        )
        run = run_agent("rust", llm, max_steps=4)
        assert run.parse_failures == 1
        assert run.answer == "Final report."

    def test_unknown_tool_is_recorded_not_raised(self):
        llm = FakeLLM([tool_call("delete_everything"), "Final report."])
        run = run_agent("rust", llm, max_steps=4)
        assert run.steps[0].ok is False
        assert "unknown tool" in run.steps[0].error
        assert run.answer == "Final report."

    def test_bad_arguments_are_recorded_not_raised(self):
        llm = FakeLLM([tool_call("check_iso_thresholds", wrong_arg=1), "Final report."])
        run = run_agent("rust", llm, max_steps=4)
        assert run.steps[0].ok is False
        assert "bad arguments" in run.steps[0].error

    def test_a_raising_tool_does_not_crash_the_run(self):
        def boom(**_kw):
            raise RuntimeError("tool exploded")

        llm = FakeLLM([tool_call("boom"), "Final report."])
        run = run_agent("rust", llm, max_steps=4, tools={"boom": boom})
        assert run.steps[0].ok is False
        assert "tool exploded" in run.steps[0].error
        assert run.answer == "Final report."

    def test_a_dead_llm_returns_an_error_answer_not_an_exception(self):
        def dead(_prompt):
            raise ConnectionError("ollama down")

        run = run_agent("rust", dead)
        assert "Report generation failed" in run.answer
        assert "ConnectionError" in run.answer


class TestAuditTrail:
    def test_records_every_tool_call_with_arguments(self):
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                tool_call("fetch_structure_history", structure_id="POLE-1"),
                "Final report.",
            ]
        )
        run = run_agent("rust", llm, max_steps=4)
        trail = run.audit_trail()
        assert trail["tool_calls"] == 2
        assert trail["trace"][0]["tool"] == "check_iso_thresholds"
        assert trail["trace"][0]["arguments"] == {"corrosion_pct": 25}
        assert trail["trace"][1]["tool"] == "fetch_structure_history"

    def test_trail_is_json_serialisable(self):
        """The audit trail is returned over HTTP, so it must serialise."""
        llm = FakeLLM([tool_call("check_iso_thresholds", corrosion_pct=25), "done"])
        run = run_agent("rust", llm)
        json.dumps(run.audit_trail())

    def test_trail_truncates_long_results(self):
        """A tool returning a huge blob must not bloat the response."""
        def verbose(**_kw):
            from src.core.agent_loop import ToolResult

            return ToolResult("verbose", {}, True, "x" * 5000)

        llm = FakeLLM([tool_call("verbose"), "done"])
        run = run_agent("rust", llm, tools={"verbose": verbose})
        assert len(run.audit_trail()["trace"][0]["result"]) <= 500


class TestAgentRunDefaults:
    def test_empty_run_has_no_tool_calls(self):
        run = AgentRun(answer="x")
        assert run.tool_calls == 0
        assert run.steps == []
        assert not run.budget_exhausted

    def test_step_defaults(self):
        s = Step(index=0, tool=None, arguments={}, ok=True, result="")
        assert s.error is None


class TestRepeatedCalls:
    """A small model re-requests the same tool when it does not like the answer.

    Observed live: three identical check_iso_thresholds calls in a row, which
    burned the entire budget and produced no new information. Re-running a pure
    tool cannot help, so the repeat is answered from cache.
    """

    def test_identical_repeat_is_answered_from_cache(self):
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                tool_call("check_iso_thresholds", corrosion_pct=25),
                "Final report.",
            ]
        )
        run = run_agent("rust", llm, max_steps=4)
        assert run.steps[1].ok
        assert "Already called" in run.steps[1].result
        assert run.answer == "Final report."

    def test_repeat_does_not_consume_a_fresh_tool_execution(self):
        """The cached step must still be recorded, so the trail stays honest."""
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                tool_call("check_iso_thresholds", corrosion_pct=25),
                "Final report.",
            ]
        )
        run = run_agent("rust", llm, max_steps=4)
        assert len(run.steps) == 2
        assert run.steps[0].result != run.steps[1].result

    def test_different_arguments_are_not_cached(self):
        llm = FakeLLM(
            [
                tool_call("check_iso_thresholds", corrosion_pct=25),
                tool_call("check_iso_thresholds", corrosion_pct=50),
                "Final report.",
            ]
        )
        run = run_agent("rust", llm, max_steps=4)
        assert "C3" in run.steps[0].result
        assert "C4" in run.steps[1].result


class TestInjectedArguments:
    """Caller-injected defaults must not collide with model-supplied arguments.

    The first live run failed every fetch_structure_history call with "got
    multiple values for keyword argument 'structure_id'", because the caller
    injected structure_id and the model supplied it too.
    """

    def test_model_argument_overrides_injected_default(self):
        llm = FakeLLM(
            [
                tool_call("fetch_structure_history", structure_id="POLE-9"),
                "Final report.",
            ]
        )
        run = run_agent(
            "rust",
            llm,
            max_steps=4,
            tool_kwargs={"fetch_structure_history": {"structure_id": "DEFAULT", "store": {}}},
        )
        assert run.steps[0].ok, run.steps[0].error
        assert "POLE-9" in run.steps[0].result

    def test_injected_default_used_when_model_omits_it(self):
        llm = FakeLLM([tool_call("fetch_structure_history"), "Final report."])
        run = run_agent(
            "rust",
            llm,
            max_steps=4,
            tool_kwargs={"fetch_structure_history": {"structure_id": "POLE-1", "store": {}}},
        )
        assert run.steps[0].ok, run.steps[0].error
        assert "POLE-1" in run.steps[0].result


class TestStructureContext:
    """The model must be told which structure it is inspecting.

    Without it the live run invented the id 12345 and fetched history for a
    structure that does not exist, which looks like a successful tool call and
    is not.
    """

    def test_structure_id_appears_in_the_prompt(self):
        llm = FakeLLM(["Final report."])
        run_agent("rust", llm, structure_id="POLE-42")
        assert "POLE-42" in llm.prompts[0]

    def test_prompt_omits_the_section_when_no_structure_is_given(self):
        llm = FakeLLM(["Final report."])
        run_agent("rust", llm)
        assert "STRUCTURE UNDER INSPECTION" not in llm.prompts[0]


class TestConfidenceIsNotAnArea:
    """A detector confidence must not be mapped to a corrosivity class.

    The live run passed 0.91 -- the rust detection confidence -- and received
    "C5, very high corrosivity". That is a fabricated engineering conclusion
    built from a number that never measured surface area, and it is exactly the
    over-claim class this project has hit before.
    """

    def test_a_confidence_shaped_value_is_rejected(self):
        r = check_iso_thresholds(0.91)
        assert not r.ok
        assert "confidence" in r.error

    def test_the_boundary_value_one_is_rejected(self):
        assert not check_iso_thresholds(1.0).ok

    def test_a_real_area_percentage_is_accepted(self):
        r = check_iso_thresholds(2.0)
        assert r.ok
        assert "C2" in r.content

    def test_zero_is_accepted_as_no_corrosion(self):
        r = check_iso_thresholds(0.0)
        assert r.ok
        assert "C1" in r.content
