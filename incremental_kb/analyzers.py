from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Analysis:
    question: str
    answer: str
    relation: str
    target_kb_id: str | None = None
    confidence: float = 1.0
    reason: str = ""


class Analyzer(Protocol):
    name: str

    def analyze(self, issue: dict, active_kb: list[dict]) -> Analysis: ...


def normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value or "").lower()


class DeterministicAnalyzer:
    """Conservative no-network analyzer used by normal ingest and tests."""

    name = "deterministic"

    def analyze(self, issue: dict, active_kb: list[dict]) -> Analysis:
        question = issue["question"].strip()
        answer = issue["answer"].strip()
        if len(normalize(question)) < 4 or len(normalize(answer)) < 4:
            return Analysis(question, answer, "low_value", reason="too_short")

        nq, na = normalize(question), normalize(answer)
        for entry in active_kb:
            if normalize(entry["question"]) != nq:
                continue
            if normalize(entry["answer"]) == na:
                return Analysis(question, answer, "duplicate_evidence", entry["kb_id"])
            conflict_words = ("不支持", "不能", "禁止", "已取消", "不可以")
            old_negative = any(word in entry["answer"] for word in conflict_words)
            new_negative = any(word in answer for word in conflict_words)
            relation = "conflict" if old_negative != new_negative else "answer_supplement"
            return Analysis(question, answer, relation, entry["kb_id"], 0.9)
        return Analysis(question, answer, "new", confidence=0.8)


class MockAnalyzer:
    name = "mock"

    def analyze(self, issue: dict, active_kb: list[dict]) -> Analysis:
        requested = issue.get("mock_relation") or "new"
        return Analysis(
            issue["question"], issue["answer"], requested,
            issue.get("mock_target_kb_id"), 1.0, "fixture",
        )


class OpenAICompatibleAnalyzer:
    name = "openai-compatible"
    allowed_relations = {
        "new", "duplicate_evidence", "answer_supplement", "multi_cause",
        "conflict", "temporal_update", "low_value",
    }

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", "")
        if not self.api_key or not self.base_url or not self.model:
            raise RuntimeError("OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL are required")

    def analyze(self, issue: dict, active_kb: list[dict]) -> Analysis:
        compact = [
            {"kb_id": x["kb_id"], "question": x["question"], "answer": x["answer"]}
            for x in active_kb
        ]
        prompt = {
            "task": "Classify the issue relation to official KB. Return strict JSON.",
            "relations": sorted(self.allowed_relations),
            "issue": {"question": issue["question"], "answer": issue["answer"]},
            "official_kb": compact,
        }
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        }).encode()
        request = Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        result = json.loads(payload["choices"][0]["message"]["content"])
        relation = result["relation"]
        if relation not in self.allowed_relations:
            raise ValueError(f"unsupported relation: {relation}")
        return Analysis(
            result.get("question") or issue["question"],
            result.get("answer") or issue["answer"], relation,
            result.get("target_kb_id"), float(result.get("confidence", 0)),
            result.get("reason", ""),
        )


def get_analyzer(name: str) -> Analyzer:
    if name == "deterministic":
        return DeterministicAnalyzer()
    if name == "mock":
        return MockAnalyzer()
    if name == "openai-compatible":
        return OpenAICompatibleAnalyzer()
    raise ValueError(f"unknown analyzer: {name}")
