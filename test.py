# src/ubs/agents/validation_agent_async.py
from __future__ import annotations

import asyncio
import json
import os
import re
from operator import add
from typing import Any, Annotated, Awaitable, Callable, Dict, List, Optional, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer  # built-in stream writer (works best on Python 3.11+)

# Optional: swap to Postgres in prod:
#   pip install langgraph-checkpoint-postgres
# from langgraph.checkpoint.postgres import PostgresSaver  # see make_checkpointer() below

try:
    from openai import AsyncAzureOpenAI  # raw SDK
except Exception:  # pragma: no cover
    AsyncAzureOpenAI = None  # allow import in environments without openai


# ---------- Types ----------
StreamFunc = Callable[[Dict[str, Any]], Awaitable[None]]

class ValidationState(TypedDict, total=False):
    # App-level context (immutable during a run unless you explicitly update it)
    context: Dict[str, Any]

    # Accumulators (reducers ensure merges are safe if you later add parallel branches)
    logs: Annotated[List[Dict[str, Any]], add]

    # Working data
    raw_input: Any
    normalized: Dict[str, Any]
    schema_errors: List[str]
    pii_hits: List[Dict[str, Any]]
    refdata_flags: List[str]
    rule_violations: List[str]
    risk_score: float
    decision: Literal["accept", "reject", "needs_review"]
    report: str


# ---------- Base agent facade (async-first) ----------
class BaseAgentAsync:
    """
    Public surface: ainvoke / astream / aresume.
    Subclasses must set self.app to a compiled LangGraph app in __init__.
    """
    name: str = "base"
    version: str = "0.0.1"

    def __init__(self, *, checkpointer: Optional[Any] = None, interrupt_before: Optional[List[str]] = None):
        self._checkpointer = checkpointer or MemorySaver()
        self._interrupt_before = interrupt_before or []
        self.app = None  # set by subclass

    # -- Execution modes
    async def ainvoke(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self.app is not None, "Agent graph not compiled"
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        return await self.app.ainvoke(initial_state, config=config)

    async def astream_events(
        self,
        initial_state: Dict[str, Any],
        *,
        thread_id: str,
        user_id: Optional[str] = None,
        modes: List[str] | None = None,
    ):
        """
        Async generator of stream events; ideal for SSE or WebSockets.
        """
        assert self.app is not None, "Agent graph not compiled"
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        modes = modes or ["updates", "custom"]  # node updates + custom events from inside nodes
        async for chunk in self.app.astream(initial_state, config, stream_mode=modes):
            yield chunk

    async def aresume(self, *, thread_id: str, resume_value: Dict[str, Any]):
        """
        Resume from an interrupt. (Not used by the linear sample below, but ready when you add HITL.)
        """
        assert self.app is not None, "Agent graph not compiled"
        from langgraph.types import Command
        config = {"configurable": {"thread_id": thread_id}}
        return await self.app.ainvoke(Command(resume=resume_value), config=config)

    # -- Helpers for subclasses
    def _compile(self, builder: StateGraph) -> Any:
        return builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self._interrupt_before,
        )


# ---------- ValidationAgent: linear workflow, async nodes ----------
class ValidationAgent(BaseAgentAsync):
    name = "validation"
    version = "1.0.0"
    REQUIRED_FIELDS = ("customer_id", "account", "amount", "currency", "timestamp")

    def __init__(
        self,
        *,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,  # deployment name in Azure, not the bare model
        checkpointer: Optional[Any] = None,
    ):
        super().__init__(checkpointer=checkpointer)
        self._oai = None
        if AsyncAzureOpenAI:
            # Raw Azure OpenAI client (no LangChain)
            # Note: in Azure, you pass the **deployment name** where OpenAI would expect a model name. :contentReference[oaicite:3]{index=3}
            self._oai = AsyncAzureOpenAI(
                api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),  # keep in sync with your tenant’s GA/preview. :contentReference[oaicite:4]{index=4}
            )
        self._deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")  # e.g., "gpt-4o-mini"

        # Build linear graph
        builder = StateGraph(ValidationState)
        builder.add_node("ingest", self._ingest)
        builder.add_node("normalize", self._normalize)
        builder.add_node("schema", self._schema)
        builder.add_node("pii", self._pii)
        builder.add_node("refdata", self._refdata)
        builder.add_node("rules", self._rules)
        builder.add_node("risk", self._risk)
        builder.add_node("summarize", self._summarize)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "ingest")
        builder.add_edge("ingest", "normalize")
        builder.add_edge("normalize", "schema")
        builder.add_edge("schema", "pii")
        builder.add_edge("pii", "refdata")
        builder.add_edge("refdata", "rules")
        builder.add_edge("rules", "risk")
        builder.add_edge("risk", "summarize")
        builder.add_edge("summarize", "finalize")
        builder.add_edge("finalize", END)

        self.app = self._compile(builder)

    # ---- Node implementations (async) ----
    async def _ingest(self, state: ValidationState, *, runtime=None, **_):
        # Prefer explicit payload in context; else try text body
        payload = state.get("context", {}).get("payload")
        if payload is None:
            payload = state.get("context", {}).get("text")
        if runtime and getattr(runtime, "stream_writer", None):
            runtime.stream_writer({"stage": "ingest", "status": "ok"})  # custom stream event :contentReference[oaicite:5]{index=5}
        return {"raw_input": payload, "logs": [{"stage": "ingest", "ok": True}]}

    async def _normalize(self, state: ValidationState, *, runtime=None, **_):
        writer = (runtime.stream_writer if runtime else get_stream_writer())
        writer and writer({"stage": "normalize", "status": "start"})
        raw = state.get("raw_input")
        out: Dict[str, Any] = {}
        if isinstance(raw, dict):
            out = raw
        elif isinstance(raw, str):
            try:
                out = json.loads(raw)
            except Exception:
                for line in raw.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        out[k.strip()] = v.strip()
        writer and writer({"stage": "normalize", "status": "end", "size": len(out)})
        return {"normalized": out}

    async def _schema(self, state: ValidationState, *, runtime=None, **_):
        d = state.get("normalized") or {}
        errs: List[str] = []
        for f in self.REQUIRED_FIELDS:
            if f not in d or d.get(f) in (None, "", []):
                errs.append(f"Missing field: {f}")
        try:
            amt = float(d.get("amount", 0))
            if not (amt > 0):
                errs.append("amount must be positive")
        except Exception:
            errs.append("amount must be numeric")
        cur = str(d.get("currency", "")).upper()
        if cur and len(cur) != 3:
            errs.append("currency must be 3 letters")
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "schema", "errors": len(errs)})
        return {"schema_errors": errs}

    async def _pii(self, state: ValidationState, *, runtime=None, **_):
        text = json.dumps(state.get("normalized", {}))
        hits: List[Dict[str, Any]] = []
        for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            hits.append({"type": "email", "value": m.group()})
        for m in re.finditer(r"\+?\d[\d \-]{7,}\d", text):
            hits.append({"type": "phone", "value": m.group()})
        for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
            hits.append({"type": "ssn_like", "value": m.group()})
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "pii", "hits": len(hits)})
        return {"pii_hits": hits}

    async def _refdata(self, state: ValidationState, *, runtime=None, **_):
        d = state.get("normalized") or {}
        flags: List[str] = []
        cur = str(d.get("currency", "")).upper()
        if cur and cur not in {"USD", "EUR", "GBP", "CHF", "JPY"}:
            flags.append(f"unknown currency: {cur}")
        acct = str(d.get("account", ""))
        if acct and not re.fullmatch(r"[A-Z0-9\-]{6,32}", acct):
            flags.append("account format looks invalid")
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "refdata", "flags": len(flags)})
        return {"refdata_flags": flags}

    async def _rules(self, state: ValidationState, *, runtime=None, **_):
        d = state.get("normalized") or {}
        violations: List[str] = []
        try:
            amt = float(d.get("amount", 0))
            if amt > 5_000_000:
                violations.append("amount exceeds 5mm auto-approval limit")
        except Exception:
            pass
        ts = str(d.get("timestamp", ""))
        if ts and not re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", ts):
            violations.append("timestamp format is not ISO-like")
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "rules", "violations": len(violations)})
        return {"rule_violations": violations}

    async def _risk(self, state: ValidationState, *, runtime=None, **_):
        score = (
            2.0 * len(state.get("schema_errors", []))
            + 1.0 * len(state.get("refdata_flags", []))
            + 1.5 * len(state.get("rule_violations", []))
            + 0.5 * len(state.get("pii_hits", []))
        )
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "risk", "score": score})
        return {"risk_score": score}

    async def _summarize(self, state: ValidationState, *, runtime=None, **_):
        """
        Optional LLM polish. Uses raw Azure OpenAI client (no LangChain).
        For strict structured output, prefer the Responses API with JSON schema
        when your SDK supports it; otherwise fall back to chat.completions. :contentReference[oaicite:6]{index=6}
        """
        d = state.get("normalized", {})
        summary = {
            "customer": d.get("customer_id"),
            "account": d.get("account"),
            "amount": d.get("amount"),
            "currency": str(d.get("currency", "")).upper(),
            "timestamp": d.get("timestamp"),
            "schema_errors": state.get("schema_errors", []),
            "refdata_flags": state.get("refdata_flags", []),
            "rule_violations": state.get("rule_violations", []),
            "pii_count": len(state.get("pii_hits", [])),
            "risk_score": state.get("risk_score", 0.0),
        }

        if not self._oai or not self._deployment:
            # No model configured; return a plain text block
            report = _render_report(summary)
            return {"report": report}

        # Try Responses API structured outputs if available for your Azure client version.
        # Some client versions expose .responses on Azure classes; otherwise use chat.completions. :contentReference[oaicite:7]{index=7}
        try:
            # Example using chat.completions with JSON-object response as a fallback
            resp = await self._oai.chat.completions.create(
                model=self._deployment,  # deployment name in Azure :contentReference[oaicite:8]{index=8}
                messages=[
                    {"role": "system", "content": "You are a precise operations assistant. Respond briefly."},
                    {"role": "user", "content": f"Turn this dict into a crisp compliance note:\n{json.dumps(summary)}"},
                ],
                response_format={"type": "json_object"},  # best-effort JSON in chat API
            )
            content = resp.choices[0].message.content or ""
            # If content is JSON, keep it; else wrap as text.
            report = content if _looks_like_json(content) else _render_report(summary)
            return {"report": report}
        except Exception:
            return {"report": _render_report(summary)}

    async def _finalize(self, state: ValidationState, *, runtime=None, **_):
        score = float(state.get("risk_score", 0.0))
        hard = bool(state.get("schema_errors"))
        soft = bool(state.get("rule_violations"))
        if hard or score >= 6.0:
            decision = "reject"
        elif soft or 3.0 <= score < 6.0:
            decision = "needs_review"
        else:
            decision = "accept"
        runtime and runtime.stream_writer and runtime.stream_writer({"stage": "finalize", "decision": decision})
        return {"decision": decision}


# ---------- Utilities ----------
def _render_report(s: Dict[str, Any]) -> str:
    lines = [
        "Validation Summary",
        "-" * 20,
        f"Customer: {s.get('customer')} | Account: {s.get('account')}",
        f"Amount: {s.get('amount')} {s.get('currency')}",
        f"Timestamp: {s.get('timestamp')}",
        "",
    ]
    def _dump(title, items):
        if items:
            lines.append(f"{title} ({len(items)}):")
            for it in items:
                lines.append(f"  - {it}")
            lines.append("")
    _dump("Schema errors", s.get("schema_errors") or [])
    _dump("Reference-data flags", s.get("refdata_flags") or [])
    _dump("Rule violations", s.get("rule_violations") or [])
    if s.get("pii_count"):
        lines.append(f"PII hits: {s['pii_count']}")
        lines.append("")
    lines.append(f"Risk score: {s.get('risk_score')}")
    return "\n".join(lines)

def _looks_like_json(t: str) -> bool:
    try:
        json.loads(t)
        return True
    except Exception:
        return False
