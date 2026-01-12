#!/usr/bin/env python3
# temporal_metrics.py
"""
Temporal Metrics Calculation Script

This script calculates temporal-specific retrieval metrics for the TEMPO benchmark,
including Temporal Recall, Temporal Precision, Temporal Relevance, and Temporal Coverage.
It supports multiple LLM providers (Azure OpenAI, OpenAI, Anthropic, Gemini) for judging temporal intent and relevance.
Data (queries, corpus, and guidance) is loaded directly from the Hugging Face 'tempo26/Tempo' dataset.

Usage:
    python temporal_metrics.py --model_results_dir <DIR> --k_values 10 --provider details

Environment Variables (depending on provider):
    Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME
    OpenAI: OPENAI_API_KEY, OPENAI_MODEL (optional, default: gpt-4o)
    Anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (optional, default: claude-3-5-sonnet-20240620)
    Gemini: GOOGLE_API_KEY, GEMINI_MODEL (optional, default: gemini-1.5-pro)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import threading
import time
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)

# =====================================================================
# PYDANTIC MODELS
# =====================================================================

class TemporalJudgmentOutput(BaseModel):
    verdict: int = Field(description="Binary verdict: 1 if temporally relevant, 0 otherwise")
    reason: str = Field(description="Explanation for the verdict")
    temporal_contribution: str = Field(description="What temporal information the document provides")


class TemporalIntentOutput(BaseModel):
    has_temporal_intent: bool = Field(description="Whether query requires temporal reasoning")
    temporal_keywords: t.List[str] = Field(description="Temporal keywords found in query")
    temporal_focus: str = Field(description="duration/specific_time/recency/change_over_time/none")


class TemporalEvidenceOutput(BaseModel):
    verdict: int = Field(description="1 if temporally relevant, else 0")
    reason: str = Field(description="brief explanation")
    temporal_contribution: str = Field(description="temporal info provided, or 'none'")
    covers_baseline: bool = Field(description="evidence about baseline anchor period")
    covers_comparison: bool = Field(description="evidence about comparison/current anchor period")


# =====================================================================
# PROMPTS
# =====================================================================

TEMPORAL_INTENT_PROMPT = """Analyze if this query requires temporal reasoning to answer correctly.

Query: "{query}"

Temporal queries ask about:
- WHEN something happened (specific time/date)
- HOW LONG something takes/lasts (duration)
- RECENT events or changes (recency)
- Changes OVER TIME (temporal evolution)
- BEFORE/AFTER relationships (temporal ordering)

Respond ONLY with valid JSON in this exact format:
{{
    "has_temporal_intent": true/false,
    "temporal_keywords": ["keyword1", "keyword2"],
    "temporal_focus": "duration" or "specific_time" or "recency" or "change_over_time" or "none"
}}

Examples:

Query: "When did Bitcoin Core introduce pruning?"
Output: {{"has_temporal_intent": true, "temporal_keywords": ["when", "introduce"], "temporal_focus": "specific_time"}}

Query: "How long does Bitcoin Core store forked chains?"
Output: {{"has_temporal_intent": true, "temporal_keywords": ["how long", "store"], "temporal_focus": "duration"}}

Query: "What are recent developments in Bitcoin storage?"
Output: {{"has_temporal_intent": true, "temporal_keywords": ["recent", "developments"], "temporal_focus": "recency"}}

Query: "What is Bitcoin Core?"
Output: {{"has_temporal_intent": false, "temporal_keywords": [], "temporal_focus": "none"}}

Now analyze: "{query}"
"""


TEMPORAL_RELEVANCE_PROMPT = """Judge if a retrieved document helps answer the TEMPORAL aspects of a query.

Query: "{query}"
Temporal Focus: {temporal_focus}

Document:
{document}

Question: Does this document provide information that DIRECTLY helps answer the temporal aspects of the query?

Guidelines:
- Verdict = 1 if document contains temporal information (dates, durations, time periods, temporal sequences)
- Verdict = 0 if document lacks temporal information even if generally relevant
- For "when" queries: document must mention specific times/dates
- For "how long" queries: document must mention durations/time periods
- For "recent" queries: document must mention recency or recent dates
- Be STRICT: generic facts without temporal markers are NOT temporally relevant

Respond ONLY with valid JSON:
{{
    "verdict": 1 or 0,
    "reason": "brief explanation",
    "temporal_contribution": "what temporal information provided, or 'none'"
}}
"""


TEMPORAL_EVIDENCE_PROMPT = """You are grading retrieved documents for a temporal trend/change query that needs cross-period evidence.

Query: "{query}"
Temporal Focus: {temporal_focus}

Baseline anchor period: {baseline_anchor}
Comparison/current anchor period: {comparison_anchor}

Document:
{document}

Decide:
1) verdict: 1 if the document DIRECTLY helps answer the temporal aspects of the query (contains relevant temporal info), else 0.
2) covers_baseline: true if the document contains evidence about the BASELINE anchor period.
3) covers_comparison: true if the document contains evidence about the COMPARISON/current anchor period.

Strictness rules:
- A random date not related to the anchors does NOT count.
- "Currently/as of 2023/recent years" can count for comparison coverage if relevant.
- Baseline coverage should connect to the baseline anchor period (e.g., around 2017).

Return ONLY valid JSON:
{{
  "verdict": 1 or 0,
  "reason": "brief explanation",
  "temporal_contribution": "what temporal information provided, or 'none'",
  "covers_baseline": true/false,
  "covers_comparison": true/false
}}
"""


# =====================================================================
# RETRY & ERROR HELPERS
# =====================================================================

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _sleep_backoff(attempt: int, base: float = 0.8, cap: float = 10.0) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay *= (0.7 + 0.6 * random.random())
    time.sleep(delay)


def _is_retryable_exception(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return int(status) in _RETRYABLE_STATUS
    msg = str(e).lower()
    return any(x in msg for x in ["rate limit", "timeout", "temporarily", "overloaded", "try again", "connection"])


# =====================================================================
# LLM Providers
# =====================================================================

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_json(self, prompt: str) -> dict:
        pass

    def generate_temporal_intent(self, query: str) -> TemporalIntentOutput:
        data = self.generate_json(TEMPORAL_INTENT_PROMPT.format(query=query))
        return TemporalIntentOutput(**data)

    def judge_temporal_relevance(self, query: str, document: str, temporal_focus: str) -> TemporalJudgmentOutput:
        doc = document[:2000] if len(document) > 2000 else document
        data = self.generate_json(
            TEMPORAL_RELEVANCE_PROMPT.format(query=query, temporal_focus=temporal_focus, document=doc)
        )
        return TemporalJudgmentOutput(**data)

    def judge_temporal_evidence(
        self,
        query: str,
        document: str,
        temporal_focus: str,
        baseline_anchor: str,
        comparison_anchor: str,
    ) -> TemporalEvidenceOutput:
        doc = document[:2000] if len(document) > 2000 else document
        data = self.generate_json(
            TEMPORAL_EVIDENCE_PROMPT.format(
                query=query,
                temporal_focus=temporal_focus,
                baseline_anchor=baseline_anchor,
                comparison_anchor=comparison_anchor,
                document=doc,
            )
        )
        return TemporalEvidenceOutput(**data)


class AzureOpenAIProvider(BaseLLMProvider):
    def __init__(self, deployment_name=None, endpoint=None, api_key=None, api_version=None, max_retries=6):
        from openai import AzureOpenAI  # Deferred import
        
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.max_retries = max_retries

        if not all([self.deployment_name, self.endpoint, self.api_key]):
            raise ValueError("Azure provider requires AZURE_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY.")
            
        self._tls = threading.local()

    def _client(self):
        from openai import AzureOpenAI
        if not hasattr(self._tls, "client"):
            self._tls.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._tls.client

    def generate_json(self, prompt: str) -> dict:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    _sleep_backoff(attempt)
                    continue
                raise
        raise RuntimeError(f"Azure call failed: {last_err}")


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None, max_retries=6):
        from openai import OpenAI  # Deferred import
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY.")

        self._tls = threading.local()

    def _client(self):
        from openai import OpenAI
        if not hasattr(self._tls, "client"):
            self._tls.client = OpenAI(api_key=self.api_key)
        return self._tls.client

    def generate_json(self, prompt: str) -> dict:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1 and _is_retryable_exception(e):
                    _sleep_backoff(attempt)
                    continue
                raise
        raise RuntimeError(f"OpenAI call failed: {last_err}")


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None, max_retries=6):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install 'anthropic' package to use Claude.")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY.")
            
        self._tls = threading.local()

    def _client(self):
        import anthropic
        if not hasattr(self._tls, "client"):
            self._tls.client = anthropic.Anthropic(api_key=self.api_key)
        return self._tls.client

    def generate_json(self, prompt: str) -> dict:
        import anthropic
        
        # Helper to extract JSON from Claude's response (since it talks a lot)
        def clean_json(text):
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return match.group(0) if match else text

        last_err = None
        for attempt in range(self.max_retries):
            try:
                # Add prefill to force JSON
                full_prompt = prompt + "\n\nImportant: Respond ONLY with valid JSON."
                
                msg = self._client().messages.create(
                    model=self.model,
                    max_tokens=1024,
                    temperature=0.0,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                content = msg.content[0].text
                return json.loads(clean_json(content))
            except Exception as e:
                last_err = e
                # Basic retry logic for Anthropic
                if attempt < self.max_retries - 1:
                    if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)):
                        _sleep_backoff(attempt)
                        continue
                raise
        raise RuntimeError(f"Anthropic call failed: {last_err}")


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None, max_retries=6):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install 'google-generativeai' package to use Gemini.")

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("Gemini provider requires GOOGLE_API_KEY.")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name, generation_config={"response_mime_type": "application/json"})

    def generate_json(self, prompt: str) -> dict:
        from google.api_core import exceptions
        
        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                return json.loads(response.text)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    # Retry on resource exhausted or server errors
                    if isinstance(e, (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.InternalServerError)):
                        _sleep_backoff(attempt)
                        continue
                raise
        raise RuntimeError(f"Gemini call failed: {last_err}")


# =====================================================================
# Metrics & Core Logic (Same as before)
# =====================================================================

@dataclass
class TemporalRecall:
    name: str = "temporal_recall"

    def calculate(self, retrieved_doc_ids: t.List[str], gold_temporal_doc_ids: t.List[str], k: int) -> float:
        if len(gold_temporal_doc_ids) == 0:
            return np.nan
        retrieved_set = set(str(d) for d in retrieved_doc_ids[:k])
        gold_set = set(str(d) for d in gold_temporal_doc_ids)
        intersection = len(retrieved_set & gold_set)
        return round(intersection / len(gold_set), 5)


def _position_weighted_precision_from_verdicts(verdicts: t.List[int]) -> float:
    relevant_positions = [i + 1 for i, v in enumerate(verdicts) if v == 1]
    if not relevant_positions:
        return 0.0
    weighted_sum = 0.0
    for rank in relevant_positions:
        relevant_up_to = sum(1 for r in relevant_positions if r <= rank)
        weighted_sum += (relevant_up_to / rank)
    return float(weighted_sum / len(relevant_positions))


def _temporal_relevance_from_verdicts(verdicts: t.List[int]) -> float:
    if not verdicts:
        return 0.0
    return float(sum(1 for v in verdicts if v == 1) / len(verdicts))


def ndcg_at_k(ranked_doc_ids: t.List[str], gold_ids: t.List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold = set(str(x) for x in gold_ids)
    if not gold:
        return np.nan

    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1.0 if str(doc_id) in gold else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return round(dcg / idcg, 5) if idcg > 0 else np.nan


def _average_metrics_from_lists(metric_lists: dict[int, dict[str, list[float]]], k_values: list[int]) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for k in k_values:
        for metric_name, vals in metric_lists[k].items():
            if vals:
                averaged[f"{metric_name}@{k}"] = round(float(np.mean(vals)), 5)
    return averaged


# =====================================================================
# HF Doc Store
# =====================================================================

class HFDocStore:
    def __init__(self, documents_dataset):
        # We assume documents_dataset is a List/Dataset where each item has 'id' and 'content'
        # To make it accessible by ID, we create a lookup
        self.doc_map = {str(d['id']): d['content'] for d in documents_dataset}

    def get(self, doc_id: str) -> str:
        return self.doc_map.get(str(doc_id), "")


# =====================================================================
# HF Guidance Loader (Steps Partition)
# =====================================================================

def load_guidance_from_steps(domain: str, cache_dir: str = None) -> dict[str, dict]:
    """
    Attempts to load query guidance from the 'steps' partition of the dataset.
    Returns a dict: {query_id: guidance_dict}
    """
    try:
        print(f"Loading guidance from 'steps' partition for {domain}...")
        ds = load_dataset("tempo26/Tempo", "steps", split=domain, cache_dir=cache_dir, trust_remote_code=True)
        guidance_map = {}
        for item in ds:
            # We assume 'id' links to query id. 
            # Note: steps partition usually has multiple rows per query (one per step).
            # We just need the guidance from ANY step of that query.
            # Assuming 'id' is Unique Query ID or there's a 'query_id' field?
            # Based on standard usage, 'id' in examples corresponds to 'id' in steps or 'qid'.
            # Inspecting user data: 'id' seems to be the key.
            # We'll overwrite duplicates (same query, same guidance usually).
            
            qid = str(item.get("id"))
            if "query_guidance" in item and item["query_guidance"]:
                guidance_map[qid] = item["query_guidance"]
        
        print(f"Loaded guidance for {len(guidance_map)} queries from steps.")
        return guidance_map
    except Exception as e:
        logger.warning(f"Failed to load steps partition for {domain}. Fallback to LLM intent detection. Error: {e}")
        return {}


# =====================================================================
# Cross-period detection + anchors
# =====================================================================

_RECENCY_WORDS = {
    "now", "nowadays", "today", "currently", "recent", "lately", "as of",
    "up-to-date", "latest", "present"
}
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def is_cross_period_query(query_guidance: dict | None, intent: TemporalIntentOutput | None, query_text: str) -> bool:
    if query_guidance and query_guidance.get("temporal_reasoning_class_primary") == "trends_changes_and_cross_period":
        return True
    if intent and intent.temporal_focus == "change_over_time":
        return True
    ql = query_text.lower()
    if "since" in ql or "over the last" in ql or "changed" in ql or "trend" in ql:
        return True
    return False


def get_anchors(query_guidance: dict | None, query_text: str) -> tuple[str, str]:
    anchors = []
    if query_guidance:
        anchors = query_guidance.get("key_time_anchors") or []
    if len(anchors) >= 2:
        return str(anchors[0]), str(anchors[1])

    years = _YEAR_RE.findall(query_text)
    baseline = years[0] if years else "historical baseline"
    ql = query_text.lower()
    comparison = "nowadays / current period" if any(w in ql for w in _RECENCY_WORDS) else "later / recent period"
    return baseline, comparison


# =====================================================================
# Main evaluation
# =====================================================================

METRIC_NAMES = [
    "temporal_recall",
    "temporal_precision",
    "temporal_relevance",
    "temporal_coverage",       # only for cross-period queries
    "ndcg",
    "ndcg_full_coverage",      # ndcg conditioned on coverage==1
]


def calculate_temporal_metrics_for_pipeline(
    model_results_dir: str,
    domains: t.List[str],
    k_values: t.List[int],
    llm: BaseLLMProvider | None,
    max_workers: int = 10,
    cache_dir: str = None
) -> dict:
    results_path = Path(model_results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    max_k = max(k_values)
    temporal_recall_metric = TemporalRecall()

    # Global aggregation
    all_metrics: dict[int, dict[str, list[float]]] = {
        k: {m: [] for m in METRIC_NAMES} for k in k_values
    }
    counts = {"total_temporal_queries": 0, "total_cross_period_queries": 0}
    domain_averages: dict[str, dict] = {}

    def eval_one_query(
        domain: str,
        query_id: str,
        query_text: str,
        gold_ids: list[str],
        retrieval_scores_for_query: dict[str, float],
        query_guidance: dict | None,
        docstore: HFDocStore,
    ) -> tuple[dict[int, dict[str, float]], bool]:
        ranked_docs = sorted(retrieval_scores_for_query.items(), key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]
        top_doc_ids = ranked_doc_ids[:max_k]

        per_k: dict[int, dict[str, float]] = {k: {} for k in k_values}

        # Always compute NDCG@k (binary on gold_ids)
        for k in k_values:
            n = ndcg_at_k(ranked_doc_ids, gold_ids, k=k)
            if not np.isnan(n):
                per_k[k]["ndcg"] = n

        # Always compute TemporalRecall@k (ID overlap on gold_ids)
        for k in k_values:
            recall = temporal_recall_metric.calculate(ranked_doc_ids, gold_ids, k=k)
            if not np.isnan(recall):
                per_k[k]["temporal_recall"] = recall

        if not llm:
            for k in k_values:
                per_k[k]["temporal_precision"] = np.nan
                per_k[k]["temporal_relevance"] = np.nan
                per_k[k]["temporal_coverage"] = np.nan
                per_k[k]["ndcg_full_coverage"] = np.nan
            return per_k, False

        # Temporal intent (gate)
        # Use guidance if available to SKIP intent generation if we trust it?
        # For now, we still let LLM confirm, OR use guidance to confirm. 
        # If guidance says it IS temporal, we trust it.
        has_guidance_intent = False
        if query_guidance and query_guidance.get("is_temporal_query"):
           has_guidance_intent = True
        
        # If we have guidance saying it's temporal, we can skip intent generation (optimization)
        # BUT we still need 'temporal_focus' for the prompt. Guidance has 'temporal_reasoning_class_primary'.
        # Let's rely on LLM for focus generation to be safe, unless we map classes.
        
        try:
            intent = llm.generate_temporal_intent(query_text)
        except Exception as e:
            logger.error(f"Error generating intent for query {query_id}: {e}")
            return per_k, False # Return empty if LLM fails

        # Hybrid check: if guidance says yes, we treat as yes. If LLM says yes, we treat as yes.
        if not (intent.has_temporal_intent or has_guidance_intent):
            for k in k_values:
                per_k[k]["temporal_precision"] = np.nan
                per_k[k]["temporal_relevance"] = np.nan
                per_k[k]["temporal_coverage"] = np.nan
                per_k[k]["ndcg_full_coverage"] = np.nan
            return per_k, False

        cross_period = is_cross_period_query(query_guidance, intent, query_text)
        baseline_anchor, comparison_anchor = get_anchors(query_guidance, query_text) if cross_period else ("", "")

        verdicts: list[int] = []
        baseline_flags: list[int] = []
        comparison_flags: list[int] = []

        for doc_id in top_doc_ids:
            doc_text = docstore.get(doc_id)
            try:
                if cross_period:
                    j = llm.judge_temporal_evidence(
                        query=query_text,
                        document=doc_text,
                        temporal_focus=intent.temporal_focus,
                        baseline_anchor=baseline_anchor,
                        comparison_anchor=comparison_anchor,
                    )
                    v = 1 if int(j.verdict) == 1 else 0
                    cb = 1 if j.covers_baseline else 0
                    cc = 1 if j.covers_comparison else 0
                else:
                    j2 = llm.judge_temporal_relevance(
                        query=query_text,
                        document=doc_text,
                        temporal_focus=intent.temporal_focus,
                    )
                    v = 1 if int(j2.verdict) == 1 else 0
                    cb, cc = 0, 0
            except Exception:
                v, cb, cc = 0, 0, 0

            verdicts.append(v)
            baseline_flags.append(cb)
            comparison_flags.append(cc)

        # LLM precision & relevance per k
        for k in k_values:
            # Slicing verdicts for top-k
            prefix_v = verdicts[:k]
            per_k[k]["temporal_relevance"] = round(_temporal_relevance_from_verdicts(prefix_v), 5)
            per_k[k]["temporal_precision"] = round(_position_weighted_precision_from_verdicts(prefix_v), 5)

        # Coverage per k (only cross-period queries)
        if cross_period:
            cum_baseline = []
            cum_comparison = []
            b = 0
            c = 0
            for i in range(len(top_doc_ids)):
                b = 1 if (b == 1 or baseline_flags[i] == 1) else 0
                c = 1 if (c == 1 or comparison_flags[i] == 1) else 0
                cum_baseline.append(b)
                cum_comparison.append(c)

            for k in k_values:
                if cum_baseline:
                    idx = min(k - 1, len(cum_baseline) - 1)
                    cov = (cum_baseline[idx] + cum_comparison[idx]) / 2.0
                else:
                    cov = 0.0
                per_k[k]["temporal_coverage"] = round(cov, 5)

                # NDCG conditioned on full coverage
                n = per_k[k].get("ndcg", np.nan)
                per_k[k]["ndcg_full_coverage"] = float(n) if (not np.isnan(n) and cov == 1.0) else np.nan
        else:
            for k in k_values:
                per_k[k]["temporal_coverage"] = np.nan
                per_k[k]["ndcg_full_coverage"] = np.nan

        return per_k, cross_period

    # ---- DOMAIN LOOP ----
    for domain in domains:
        logger.info("Processing domain: %s", domain)

        # Per-domain aggregation
        domain_metrics: dict[int, dict[str, list[float]]] = {
            k: {m: [] for m in METRIC_NAMES} for k in k_values
        }
        domain_counts = {"total_temporal_queries": 0, "total_cross_period_queries": 0}

        scores_file = results_path / domain / "scores.json"
        if not scores_file.exists():
            logger.warning("Scores file not found: %s", scores_file)
            continue

        try:
            with scores_file.open("r", encoding="utf-8") as f:
                retrieval_scores = json.load(f)
        except Exception as e:
             logger.error(f"Error loading scores file for {domain}: {e}")
             continue

        # Load data from HuggingFace
        try:
            print(f"Loading queries for {domain} from HF...")
            queries_ds = load_dataset("tempo26/Tempo", "examples", split=domain, cache_dir=cache_dir)
            
            # Load guidance from steps (optional, fail-safe)
            raw_guidance_map = load_guidance_from_steps(domain, cache_dir=cache_dir)

            print(f"Loading corpus for {domain} from HF...")
            docs_ds = load_dataset("tempo26/Tempo", "documents", split=domain, cache_dir=cache_dir)
            docstore = HFDocStore(docs_ds)
            
        except Exception as e:
            logger.error(f"Error loading HF datasets for {domain}: {e}")
            continue

        # Filter query IDs that we actually have scores for
        tasks = []
        for q in queries_ds:
            qid = str(q["id"])
            if qid not in retrieval_scores:
                if int(qid) in retrieval_scores:
                    qid = int(qid)
                elif str(qid) in retrieval_scores:
                    qid = str(qid)
                else:
                    continue
            
            # Look up guidance if available
            qguid = raw_guidance_map.get(qid)
            
            tasks.append(
                (
                    domain,
                    str(qid),
                    q["query"],
                    q.get("gold_ids", []),
                    retrieval_scores[qid],
                    qguid,
                )
            )

        if not tasks:
            logger.warning(f"No matching queries found in scores for {domain}.")
            continue

        counts["total_temporal_queries"] += len(tasks)
        domain_counts["total_temporal_queries"] += len(tasks)

        # Ensure domain results folder exists
        (results_path / domain).mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    eval_one_query,
                    domain,
                    qid,
                    qtext,
                    gold_ids,
                    retrieval_scores_for_query,
                    qguid,
                    docstore,
                )
                for (domain, qid, qtext, gold_ids, retrieval_scores_for_query, qguid) in tasks
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{domain}: temporal queries"):
                try:
                    per_k, cross_period = fut.result()
                except Exception as e:
                    logger.error(f"Error processing query in {domain}: {e}")
                    continue

                if cross_period:
                    counts["total_cross_period_queries"] += 1
                    domain_counts["total_cross_period_queries"] += 1

                for k in k_values:
                    for m in METRIC_NAMES:
                        v = per_k[k].get(m, np.nan)
                        if not np.isnan(v):
                            all_metrics[k][m].append(float(v))
                            domain_metrics[k][m].append(float(v))

        # Save per-domain averages
        domain_avg = _average_metrics_from_lists(domain_metrics, k_values)
        per_domain_payload = {
            "domain": domain,
            "averaged_metrics": domain_avg,
            "counts": domain_counts,
            "k_values": k_values,
            "max_workers": max_workers,
        }
        per_domain_path = results_path / domain / "temporal_metrics.json"
        with per_domain_path.open("w", encoding="utf-8") as f:
            json.dump(per_domain_payload, f, indent=2, ensure_ascii=False)

        domain_averages[domain] = per_domain_payload

    # Global averages and summary file
    averaged_metrics = _average_metrics_from_lists(all_metrics, k_values)
    summary_payload = {
        "averaged_metrics": averaged_metrics,
        "counts": counts,
        "per_domain": domain_averages,
        "k_values": k_values,
        "max_workers": max_workers,
        "domains": domains,
    }

    summary_path = results_path / "temporal_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    return summary_payload


# =====================================================================
# CLI
# =====================================================================

DEFAULT_DOMAINS = [
    "bitcoin",
    "cardano",
    "economics",
    "genealogy",
    "history",
    "hsm",
    "iota",
    "law",
    "monero",
    "politics",
    "quant",
    "travel",
    "workplace",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temporal retrieval metrics with multi-provider LLM support, loading data from HuggingFace."
    )
    p.add_argument("--model_results_dir", required=True, type=str, help="Directory containing model result folders.")
    p.add_argument("--cache_dir", default="cache", type=str, help="Directory for caching HF datasets.")
    p.add_argument("--k_values", nargs="+", type=int, required=True, help="List of K values for metrics (e.g., 10 20).")
    p.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, help="List of domains to evaluate.")

    p.add_argument("--provider", type=str, choices=["azure", "openai", "anthropic", "gemini"], default="azure", 
                   help="LLM provider to use for temporal judgment. Default: azure")
    
    p.add_argument("--no_llm", action="store_true", help="Disable all LLM-based metrics.")
    
    p.add_argument("--max_workers", type=int, default=10, help="Number of worker threads.")

    # Azure specific
    p.add_argument("--azure_endpoint", type=str, default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    p.add_argument("--azure_api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    p.add_argument("--api_version", type=str, default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"))
    p.add_argument("--deployment_name", type=str, default=os.getenv("AZURE_DEPLOYMENT_NAME"))

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    llm = None
    if not args.no_llm:
        try:
            if args.provider == "azure":
                llm = AzureOpenAIProvider(
                    deployment_name=args.deployment_name,
                    endpoint=args.azure_endpoint,
                    api_key=args.azure_api_key,
                    api_version=args.api_version
                )
            elif args.provider == "openai":
                llm = OpenAIProvider()
            elif args.provider == "anthropic":
                llm = AnthropicProvider()
            elif args.provider == "gemini":
                llm = GeminiProvider()
            
            logging.info(f"Initialized {args.provider.upper()} provider successfully.")
        except (ValueError, ImportError) as e:
            logging.error(f"Failed to initialize {args.provider} LLM: {e}")
            logging.error("Please set the required environment variables or check dependencies.")
            return

    results = calculate_temporal_metrics_for_pipeline(
        model_results_dir=args.model_results_dir,
        cache_dir=args.cache_dir,
        domains=args.domains,
        k_values=args.k_values,
        llm=llm,
        max_workers=args.max_workers,
    )

    print(json.dumps(results["averaged_metrics"], indent=2, ensure_ascii=False))
    print(f"\nSaved per-domain metrics to: {args.model_results_dir}/<domain>/temporal_metrics.json")
    print(f"Saved global summary to:    {args.model_results_dir}/temporal_metrics_summary.json")


if __name__ == "__main__":
    main()
