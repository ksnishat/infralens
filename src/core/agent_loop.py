"""Bounded tool-calling loop for the InfraLens safety-report agent.

The previous agent was a single prompt and a single LLM call. It could not look
anything up, so every report was written from the detection text alone. Calling
that an agent would not survive a follow-up question: there was no tool
selection, no iteration, and no record of what the model did.

This module adds an explicit loop:

    model asks for a tool -> tool runs -> result is fed back -> repeat

with a hard step budget. No framework. The loop is ~120 lines and every branch
is testable without a model.

Design decisions worth defending
--------------------------------
* **Bounded, not open-ended.** `max_steps` defaults to 4. On exhaustion the loop
  returns the best partial answer *and says so* in the audit trail, rather than
  silently returning whatever the model last said.
* **Tools are pure functions over local data.** No network, no writes. A tool
  that mutates state inside a retry loop is a bug waiting to happen.
* **The audit trail is the product.** Every tool call, its arguments, its
  result and the step count are recorded. That is what makes the loop
  inspectable, and it is the part an interviewer can actually check.
* **Malformed tool JSON is a normal outcome, not an exception.** A small local
  model emits invalid JSON regularly. The loop records the parse failure and
  continues rather than crashing the request.

On the EU AI Act: this is *not* a compliant high-risk system and the code does
not claim to be. The audit trail is shaped like the record-keeping Article 12
would require, as an exercise in AI governance. A portfolio demo is not a
deployed safety component, and claiming otherwise would not survive scrutiny.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4

# DIN EN ISO 12944 atmospheric corrosivity categories, by mass loss per unit
# area over the first year of exposure (g/m^2), for low-carbon steel.
# Source: DIN EN ISO 12944-2, Table 1.
ISO_12944_CATEGORIES: tuple[tuple[str, float, float, str], ...] = (
    ("C1", 0.0, 10.0, "very low"),
    ("C2", 10.0, 200.0, "low"),
    ("C3", 200.0, 400.0, "medium"),
    ("C4", 400.0, 650.0, "high"),
    ("C5", 650.0, 1500.0, "very high"),
    ("CX", 1500.0, float("inf"), "extreme"),
)


# --------------------------------------------------------------------- tools


@dataclass
class ToolResult:
    """What a tool returned, in a shape the loop can feed back to the model."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    content: str
    error: str | None = None


def check_iso_thresholds(corrosion_pct: float) -> ToolResult:
    """Map a corrosion percentage to a DIN EN ISO 12944 corrosivity category.

    The mapping is a documented approximation, not a measurement: ISO 12944
    classifies by mass loss per unit area, and a detector reports a fraction of
    the image. Treating the two as equivalent is the kind of quiet
    over-claim this project has been bitten by before, so the tool says which
    one it is doing.
    """
    try:
        pct = float(corrosion_pct)
    except (TypeError, ValueError):
        return ToolResult(
            name="check_iso_thresholds",
            arguments={"corrosion_pct": corrosion_pct},
            ok=False,
            content="",
            error=f"corrosion_pct must be a number, got {corrosion_pct!r}",
        )

    if pct < 0 or pct > 100:
        return ToolResult(
            name="check_iso_thresholds",
            arguments={"corrosion_pct": pct},
            ok=False,
            content="",
            error=f"corrosion_pct must be between 0 and 100, got {pct}",
        )

    # A detector confidence is a model certainty, not a surface-area estimate.
    # The live run passed 0.91 (a confidence) and got C5 "extreme corrosivity",
    # which is a fabricated engineering conclusion from a number that never
    # measured area. Refuse values in that range rather than map them.
    if 0 < pct <= 1:
        return ToolResult(
            name="check_iso_thresholds",
            arguments={"corrosion_pct": pct},
            ok=False,
            content="",
            error=(
                f"corrosion_pct={pct} looks like a detection confidence, not a "
                "surface-area percentage. Pass the estimated affected area "
                "(0-100), or omit this tool."
            ),
        )

    # Approximate: 1% of surface area is treated as 10 g/m^2 of mass loss.
    mass_loss = pct * 10.0
    for name, low, high, label in ISO_12944_CATEGORIES:
        if low <= mass_loss < high:
            return ToolResult(
                name="check_iso_thresholds",
                arguments={"corrosion_pct": pct},
                ok=True,
                content=(
                    f"corrosion_pct={pct} -> approx {mass_loss:.0f} g/m^2 "
                    f"-> ISO 12944 category {name} ({label} corrosivity). "
                    "Approximation: ISO classifies by mass loss, the detector "
                    "reports image area fraction."
                ),
            )

    return ToolResult(
        name="check_iso_thresholds",
        arguments={"corrosion_pct": pct},
        ok=False,
        content="",
        error="no category matched",
    )


def fetch_structure_history(structure_id: str, store: dict | None = None) -> ToolResult:
    """Return prior inspection records for a structure.

    Backed by an in-memory store by default. The project has no inspection
    history table -- `src/app/db/models.py` defines Tenant, User and Asset only,
    and the Postgres instance is not part of the default stack -- so inventing a
    database dependency here would make the tool untestable and the claim false.
    The store is injectable so a real repository can be passed in later without
    changing the tool's contract.
    """
    records = (store or {}).get(str(structure_id), [])
    if not records:
        return ToolResult(
            name="fetch_structure_history",
            arguments={"structure_id": structure_id},
            ok=True,
            content=(
                f"No prior inspections recorded for structure {structure_id!r}. "
                "Treat this as a first inspection."
            ),
        )
    lines = [
        f"- {r.get('date', 'unknown date')}: {r.get('finding', 'no finding')} "
        f"(severity {r.get('severity', 'n/a')})"
        for r in records
    ]
    return ToolResult(
        name="fetch_structure_history",
        arguments={"structure_id": structure_id},
        ok=True,
        content=f"Prior inspections for {structure_id!r}:\n" + "\n".join(lines),
    )


TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "check_iso_thresholds": check_iso_thresholds,
    "fetch_structure_history": fetch_structure_history,
}

TOOL_SCHEMAS = [
    {
        "name": "check_iso_thresholds",
        "description": (
            "Map the estimated percentage of a surface affected by corrosion "
            "to a DIN EN ISO 12944 corrosivity category (C1..CX). "
            "corrosion_pct is the estimated SURFACE AREA PERCENTAGE affected, "
            "not a detector confidence score. If only a confidence score is "
            "known, use the affected area estimate instead, or omit this call."
        ),
        "parameters": {
            "corrosion_pct": (
                "estimated percentage of surface affected by corrosion, 0-100. "
                "NOT the detection confidence (which is a model certainty, not "
                "an area)."
            )
        },
    },
    {
        "name": "fetch_structure_history",
        "description": "Return prior inspection records for a structure id.",
        "parameters": {"structure_id": "string"},
    },
]


# ---------------------------------------------------------------------- loop


@dataclass
class Step:
    """One iteration of the loop, recorded for the audit trail."""

    index: int
    tool: str | None
    arguments: dict[str, Any]
    ok: bool
    result: str
    error: str | None = None


@dataclass
class AgentRun:
    """The outcome of one bounded run."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    budget_exhausted: bool = False
    parse_failures: int = 0

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool)

    def audit_trail(self) -> dict[str, Any]:
        """The record an auditor would read.

        Deliberately includes the budget flag and the parse-failure count: a
        run that hit the budget produced a partial answer, and a reader must be
        able to tell that from a complete one.
        """
        return {
            "steps": len(self.steps),
            "tool_calls": self.tool_calls,
            "budget_exhausted": self.budget_exhausted,
            "parse_failures": self.parse_failures,
            "trace": [
                {
                    "step": s.index,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "ok": s.ok,
                    "result": s.result[:500],
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


def _extract_json_objects(text: str) -> list[str]:
    r"""Return every balanced {...} block in `text`, outermost first.

    A regex cannot do this: a tool call contains a nested `arguments` object, so
    `\{[^{}]*\}` matches nothing. Scanning with a depth counter handles
    nesting and also survives a truncated object, which is what a small model
    emits when it runs out of tokens mid-call.
    """
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    blocks.append(text[start : i + 1])
                    start = -1
    return blocks


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool call from model output.

    Returns None when the text contains no tool call, which is the normal way a
    model signals it is ready to answer. Raises ValueError only when a tool call
    is present but malformed, so the caller can count parse failures separately
    from "no call".
    """
    blocks = _extract_json_objects(text)
    if not blocks:
        # An unbalanced brace means the model started a call and was cut off.
        if "{" in text and '"tool"' in text:
            raise ValueError("malformed tool call: unbalanced braces")
        return None

    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed tool call: {exc}") from exc
        if isinstance(payload, dict) and "tool" in payload:
            return payload

    # JSON was present but none of it was a tool call. A dict carrying
    # `arguments` is clearly an attempted call with the key misspelled or
    # omitted, so it is a parse failure rather than a final answer.
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "arguments" in payload:
            raise ValueError("tool call missing 'tool' key")
    return None


def _build_prompt(
    detection_data: str,
    history: list[Step],
    schemas: list[dict],
    structure_id: str | None = None,
) -> str:
    """Build the loop prompt, including everything the model has learned so far."""
    tool_docs = json.dumps(schemas, indent=2)
    if history:
        transcript = "\n".join(
            f"Step {s.index}: called {s.tool}({json.dumps(s.arguments)}) -> {s.result}"
            for s in history
        )
    else:
        transcript = "(no tools called yet)"

    subject = (
        f"\nSTRUCTURE UNDER INSPECTION: {structure_id}\n"
        "Use this exact id when calling fetch_structure_history.\n"
        if structure_id
        else ""
    )

    return f"""You are an industrial safety inspector for a German facility.
{subject}
DETECTION DATA:
{detection_data}

AVAILABLE TOOLS:
{tool_docs}

TOOL CALLS SO FAR:
{transcript}

If you need more information, reply with ONLY a JSON object:
{{"tool": "<name>", "arguments": {{...}}}}

If you have enough information, write the final safety report. Format:
1. Summary of findings
2. Severity assessment (LOW / MEDIUM / HIGH / CRITICAL)
3. Root cause analysis (brief)
4. Recommended action (with timeline)
5. Compliance reference (DIN EN ISO 12944 / DGUV Vorschrift 3)

Write in German and English. Be precise and factual. Do not invent figures."""


def run_agent(
    detection_data: str,
    llm: Callable[[str], str],
    max_steps: int = DEFAULT_MAX_STEPS,
    tools: dict[str, Callable[..., ToolResult]] | None = None,
    tool_kwargs: dict[str, dict] | None = None,
    structure_id: str | None = None,
) -> AgentRun:
    """Run the bounded tool loop.

    `llm` is any callable taking a prompt and returning text. Injecting it keeps
    the loop testable without a model, which is the only way to test the
    exhaustion and malformed-JSON paths deterministically.
    """
    registry = tools if tools is not None else TOOL_REGISTRY
    extra = tool_kwargs or {}
    steps: list[Step] = []
    parse_failures = 0
    seen: dict[tuple[str, str], str] = {}

    for i in range(max_steps):
        prompt = _build_prompt(detection_data, steps, TOOL_SCHEMAS, structure_id)
        try:
            raw = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - a dead model must not 500
            return AgentRun(
                answer=f"Report generation failed: {type(exc).__name__}: {exc}",
                steps=steps,
                budget_exhausted=False,
                parse_failures=parse_failures,
            )

        try:
            call = parse_tool_call(raw)
        except ValueError as exc:
            parse_failures += 1
            steps.append(
                Step(
                    index=i,
                    tool=None,
                    arguments={},
                    ok=False,
                    result="",
                    error=str(exc),
                )
            )
            # Feed the failure back so the model can correct itself.
            continue

        if call is None:
            # No tool requested: this is the final answer.
            return AgentRun(
                answer=raw.strip(),
                steps=steps,
                budget_exhausted=False,
                parse_failures=parse_failures,
            )

        name = call.get("tool")
        args = call.get("arguments") or {}

        # A small model re-requests the same tool with the same arguments when
        # it does not recognise the result as an answer. Observed live: three
        # identical check_iso_thresholds calls in a row, burning the whole
        # budget. Re-running a pure tool cannot produce new information, so the
        # repeat is answered from the cache and the model is told so.
        cache_key = (str(name), json.dumps(args, sort_keys=True))
        if cache_key in seen:
            steps.append(
                Step(
                    index=i,
                    tool=str(name),
                    arguments=args if isinstance(args, dict) else {},
                    ok=True,
                    result=(
                        f"Already called with these arguments. Result: {seen[cache_key]}"
                    ),
                )
            )
            continue

        fn = registry.get(name)
        if fn is None:
            steps.append(
                Step(
                    index=i,
                    tool=str(name),
                    arguments=args if isinstance(args, dict) else {},
                    ok=False,
                    result="",
                    error=f"unknown tool {name!r}; available: {sorted(registry)}",
                )
            )
            continue

        # Model-supplied arguments win over injected defaults. Merging the
        # other way round raised "got multiple values for keyword argument"
        # whenever the model supplied a key the caller had also injected --
        # which happened on the first live run.
        merged = {**extra.get(name, {}), **(args if isinstance(args, dict) else {})}
        try:
            result = fn(**merged)
        except TypeError as exc:
            steps.append(
                Step(
                    index=i,
                    tool=str(name),
                    arguments=args if isinstance(args, dict) else {},
                    ok=False,
                    result="",
                    error=f"bad arguments: {exc}",
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 - a tool bug must not 500
            steps.append(
                Step(
                    index=i,
                    tool=str(name),
                    arguments=args if isinstance(args, dict) else {},
                    ok=False,
                    result="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if result.ok:
            seen[cache_key] = result.content
        steps.append(
            Step(
                index=i,
                tool=result.name,
                arguments=result.arguments,
                ok=result.ok,
                result=result.content,
                error=result.error,
            )
        )

    # Budget exhausted. Ask once more for a final answer with no tools, so the
    # caller gets prose rather than a tool call it cannot act on.
    final_prompt = _build_prompt(detection_data, steps, TOOL_SCHEMAS, structure_id) + (
        "\n\nYou have used all available tool calls. Write the final report now "
        "using what you have. State any limitation caused by missing information."
    )
    try:
        answer = llm(final_prompt).strip()
    except Exception as exc:  # noqa: BLE001
        answer = f"Report generation failed after budget exhaustion: {exc}"

    return AgentRun(
        answer=answer,
        steps=steps,
        budget_exhausted=True,
        parse_failures=parse_failures,
    )


__all__ = [
    "AgentRun",
    "Step",
    "ToolResult",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "check_iso_thresholds",
    "fetch_structure_history",
    "parse_tool_call",
    "run_agent",
]
