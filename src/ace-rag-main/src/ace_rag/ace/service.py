from __future__ import annotations

import re
import time
from typing import Any

from ace_rag.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    PlaybookOrganizeRequest,
    PlaybookOrganizeResponse,
    PlaybookDebug,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    RoutePlanResponse,
    EvidenceItem,
    LLMConfig,
)
from ace_rag.core.config import get_settings, pydantic_dump
from ace_rag.core.text import dedupe_keep_order, infer_query_intents, merge_sources, normalize_query
from ace_rag.llm.client import answer_with_llm, organize_playbook_with_llm
from ace_rag.playbook.seed import load_seed_items
from ace_rag.playbook.store import PlaybookStore, collect_expansion_terms, collect_source_hints
from ace_rag.v2_client.client import V2Client


SENSITIVE_KEYS = {"api_key", "authorization", "token", "password", "secret"}


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                redacted[key] = "***"
            else:
                redacted[key] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


class AceService:
    def __init__(self, store: PlaybookStore, v2_client: V2Client, *, seed_path=None, auto_import_seed: bool = True) -> None:
        self.store = store
        self.v2_client = v2_client
        if auto_import_seed and self.store.get_item_count() == 0:
            self.store.import_items(load_seed_items(seed_path))
        self.store.ensure_auto_organize_baseline()

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        started = time.perf_counter()
        playbook_started = time.perf_counter()
        intents = infer_query_intents(request.query)
        playbook_items = (
            self.store.search(request.query, top_k=request.playbook_top_k, intents=intents)
            if request.use_playbook
            else []
        )
        playbook_ms = round((time.perf_counter() - playbook_started) * 1000, 3)

        expansion_terms = collect_expansion_terms(
            playbook_items,
            request.query_expansion_terms,
            query=request.query,
        )
        source_hints = collect_source_hints(playbook_items)
        sources = merge_sources(request.sources, source_hints)
        v2_request = self._build_v2_retrieve_request(
            request=request,
            sources=sources,
            expansion_terms=expansion_terms,
        )

        playbook_debug = PlaybookDebug(
            used=request.use_playbook,
            items=playbook_items,
            query_expansion_terms=expansion_terms,
            source_hints=source_hints,
        )
        if request.playbook_only:
            return RetrieveResponse(
                query=request.query,
                route_plan=RoutePlanResponse.model_validate(self._fallback_route_plan(sources)),
                evidences=[],
                timings={
                    "playbook_retrieve_ms": playbook_ms,
                    "v2_retrieve_ms": 0.0,
                    "ace_total_ms": round((time.perf_counter() - started) * 1000, 3),
                    "playbook_only": True,
                },
                retrieved_count=0,
                playbook=playbook_debug,
                v2_request={
                    **v2_request,
                    "_skipped": True,
                    "_skip_reason": "playbook_only",
                },
            )

        v2_started = time.perf_counter()
        v2_response = await self.v2_client.retrieve(v2_request)
        v2_ms = round((time.perf_counter() - v2_started) * 1000, 3)

        evidences = [EvidenceItem.model_validate(item) for item in v2_response.get("evidences") or []]
        route_plan = RoutePlanResponse.model_validate(v2_response.get("route_plan") or self._fallback_route_plan(sources))
        timings = dict(v2_response.get("timings") or {})
        timings.update(
            {
                "playbook_retrieve_ms": playbook_ms,
                "v2_retrieve_ms": v2_ms,
                "ace_total_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return RetrieveResponse(
            query=str(v2_response.get("query") or request.query),
            route_plan=route_plan,
            evidences=evidences,
            timings=timings,
            retrieved_count=int(v2_response.get("retrieved_count") or len(evidences)),
            playbook=playbook_debug,
            v2_request=v2_request,
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        retrieve_response = await self.retrieve(request)
        timings = dict(retrieve_response.timings)
        answer_started = time.perf_counter()
        override = self._online_feedback_override(request.query, retrieve_response)
        if override is not None:
            answer, llm_config = override
            timings["online_feedback_override_ms"] = 0.0
        else:
            v2_answer = await self._v2_query_answer(
                v2_request=retrieve_response.v2_request,
                llm_config=request.llm_config,
                timings=timings,
            )
            if v2_answer is not None:
                answer, llm_config = v2_answer
            else:
                deterministic_answer = self._deterministic_v2_answer(retrieve_response)
                if deterministic_answer is not None:
                    answer = deterministic_answer
                    llm_config = (LLMConfig().model_dump())
                    llm_config.update({"mode": "v2_deterministic_answer"})
                else:
                    answer, llm_config = await answer_with_llm(
                        query=request.query,
                        evidences=retrieve_response.evidences,
                        playbook_items=retrieve_response.playbook.items,
                        llm_config=request.llm_config,
                    )
        timings["answer_ms"] = round((time.perf_counter() - answer_started) * 1000, 3)
        timings["ace_query_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        request_payload = _redact_sensitive(pydantic_dump(request))
        v2_response_payload = {
            "query": retrieve_response.query,
            "route_plan": retrieve_response.route_plan.model_dump(),
            "evidences": [item.model_dump() for item in retrieve_response.evidences],
            "timings": retrieve_response.timings,
            "retrieved_count": retrieve_response.retrieved_count,
        }
        run_id = self.store.insert_run(
            query=request.query,
            request=request_payload,
            v2_request=retrieve_response.v2_request,
            v2_response=v2_response_payload,
            playbook_item_ids=[item.item_id for item in retrieve_response.playbook.items],
            answer=answer,
            timings=timings,
        )
        response_payload = retrieve_response.model_dump()
        response_payload["timings"] = timings
        return QueryResponse(
            **response_payload,
            answer=answer,
            llm_config=_redact_sensitive(llm_config),
            run_id=run_id,
        )

    async def _v2_query_answer(
        self,
        *,
        v2_request: dict[str, Any],
        llm_config: LLMConfig | None,
        timings: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        payload = dict(v2_request)
        if llm_config is not None:
            payload["llm_config"] = llm_config.model_dump()
        started = time.perf_counter()
        try:
            response = await self.v2_client.query(payload)
        except Exception as exc:
            timings["v2_query_error"] = str(exc)
            timings["v2_query_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return None
        timings["v2_query_ms"] = round((time.perf_counter() - started) * 1000, 3)
        v2_timings = response.get("timings")
        if isinstance(v2_timings, dict):
            timings["v2_query_timings"] = v2_timings
        answer = str(response.get("answer") or "").strip()
        if not answer:
            return None
        config = response.get("llm_config")
        if not isinstance(config, dict):
            config = (LLMConfig().model_dump())
        config.update({"mode": "gbrain_v2_query_answer"})
        return answer, config

    def _deterministic_v2_answer(self, retrieve_response: RetrieveResponse) -> str | None:
        for idx, evidence in enumerate(retrieve_response.evidences, start=1):
            answer = str(evidence.payload.get("deterministic_answer") or "").strip()
            if not answer:
                continue
            if re.search(r"\[证据\d+\]", answer):
                return answer
            return f"{answer.rstrip()} [证据{idx}]"
        return None

    async def add_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        result = self.store.insert_feedback_detailed(request)
        await self._maybe_auto_organize_playbook(
            feedback_id=result.feedback_id,
            online_item_id=result.online_item_id,
        )
        return FeedbackResponse(
            feedback_id=result.feedback_id,
            operation_id=result.operation_id,
            status="pending",
        )

    def organize_playbook(self, request: PlaybookOrganizeRequest) -> PlaybookOrganizeResponse:
        item_count, candidates = self.store.organize_candidates(
            include_sections=request.include_sections,
            include_inactive=request.include_inactive,
            min_confidence=request.min_confidence,
            max_items=request.max_items,
        )
        return PlaybookOrganizeResponse(
            item_count=item_count,
            strategies=[
                {
                    "name": "semantic_merge_and_deduplicate",
                    "description": "按 section、query_intents、source_hints、tags 将相近规则聚类，生成合并候选，适合 seed 规则和人工规则去重。",
                },
                {
                    "name": "episodic_feedback_to_semantic_rule",
                    "description": "把 online_feedback 中的单次纠错记忆抽象为可复用规则，保留原 feedback/run provenance 供人工审核。",
                },
                {
                    "name": "single_item_summary",
                    "description": "对独立规则生成短摘要、关键扩展词和适用范围，适合作为只读 memory index 或后续人工编辑入口。",
                },
            ],
            candidates=candidates,
        )

    def _build_v2_retrieve_request(
        self,
        *,
        request: RetrieveRequest,
        sources: list[str] | None,
        expansion_terms: list[str],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": request.query,
            "retrieval_method": request.retrieval_method,
            "top_k": request.top_k,
            "candidate_limit": request.candidate_limit,
            "route_with_llm": request.route_with_llm,
            "expand_query_with_llm": request.expand_query_with_llm,
            "query_expansion_terms": expansion_terms,
            "document": request.document.model_dump(),
            "table": request.table.model_dump(),
            "adela": request.adela.model_dump(),
        }
        optional_values = {
            "similarity_threshold": request.similarity_threshold,
            "sources": sources,
            "embedding_model": request.embedding_model,
            "embedding_models": request.embedding_models,
            "embedding_backend": request.embedding_backend,
        }
        for key, value in optional_values.items():
            if value not in (None, [], {}):
                payload[key] = value
        return payload

    async def _maybe_auto_organize_playbook(self, *, feedback_id: str, online_item_id: str | None) -> str | None:
        if not online_item_id:
            return None
        settings = get_settings()
        if not settings.ENABLE_PLAYBOOK_AUTO_ORGANIZE:
            return None

        threshold = max(int(settings.PLAYBOOK_ORGANIZE_DELTA_THRESHOLD), 1)
        last_count = self.store.get_auto_organize_baseline_count()
        if last_count is None:
            last_count = self.store.ensure_auto_organize_baseline()
        current_count = self.store.get_item_count()
        delta = current_count - last_count
        if delta < threshold:
            return None

        items = self._select_items_for_auto_organization(settings.PLAYBOOK_ORGANIZE_MAX_ITEMS)
        new_items, retire_item_ids, metadata = await organize_playbook_with_llm(items=items)
        preserved_retire_ids = [item_id for item_id in retire_item_ids if item_id == online_item_id]
        retire_item_ids = [item_id for item_id in retire_item_ids if item_id != online_item_id]
        payload: dict[str, Any] = {
            "trigger": "online_feedback",
            "trigger_feedback_id": feedback_id,
            "trigger_online_item_id": online_item_id,
            "preserved_trigger_online_item_ids": preserved_retire_ids,
            "threshold": threshold,
            "baseline_active_item_count": last_count,
            "current_active_item_count_before_apply": current_count,
            "delta": delta,
            **metadata,
        }
        if not new_items and not retire_item_ids:
            status = "skipped" if metadata.get("mode") in {"llm_disabled", "llm_error"} else "noop"
            return self.store.record_auto_organization_attempt(
                feedback_id=feedback_id,
                status=status,
                payload=payload,
                update_baseline=True,
            )
        return self.store.apply_auto_organization(
            feedback_id=feedback_id,
            new_items=new_items,
            retire_item_ids=retire_item_ids,
            payload=payload,
        )

    def _select_items_for_auto_organization(self, max_items: int) -> list[Any]:
        items = self.store.list_items(include_inactive=False)
        if max_items <= 0 or len(items) <= max_items:
            return items

        online_feedback = [item for item in items if item.section == "online_feedback"]
        online_feedback.sort(key=lambda item: item.updated_at or 0, reverse=True)
        remaining_slots = max(max_items - len(online_feedback), 0)
        stable_items = [item for item in items if item.section != "online_feedback"]
        stable_items.sort(key=lambda item: (item.confidence, item.updated_at or 0), reverse=True)
        return [*online_feedback[:max_items], *stable_items[:remaining_slots]][:max_items]

    def _fallback_route_plan(self, sources: list[str] | None) -> dict[str, Any]:
        resolved = sources or ["document", "table", "adela"]
        return {
            "document": "document" in resolved,
            "table": "table" in resolved,
            "adela": "adela" in resolved,
            "reason": "v2 response did not include route_plan; generated by ace-rag fallback",
            "sources": resolved,
        }

    def _online_feedback_override(
        self,
        query: str,
        retrieve_response: RetrieveResponse,
    ) -> tuple[str, dict[str, Any]] | None:
        normalized_query = normalize_query(query)
        for item in retrieve_response.playbook.items:
            if item.section != "online_feedback":
                continue
            source_query = normalize_query(str(item.provenance.get("source_query") or ""))
            corrected_answer = str(item.provenance.get("corrected_answer") or "").strip()
            if not corrected_answer or not source_query or source_query != normalized_query:
                continue
            answer = self._answer_with_expected_citations(
                corrected_answer,
                retrieve_response.evidences,
                item.provenance.get("expected_evidence_ids") or [],
            )
            llm_config = (LLMConfig().model_dump())
            llm_config.update(
                {
                    "mode": "online_feedback_override",
                    "applied_feedback_id": item.provenance.get("feedback_id"),
                    "source_query": item.provenance.get("source_query"),
                }
            )
            return answer, llm_config
        return None

    def _answer_with_expected_citations(
        self,
        answer: str,
        evidences: list[EvidenceItem],
        expected_evidence_ids: list[str],
    ) -> str:
        if not expected_evidence_ids or re.search(r"\[证据\d+\]", answer):
            return answer
        citations: list[str] = []
        expected = {str(value) for value in expected_evidence_ids if str(value)}
        for idx, evidence in enumerate(evidences, start=1):
            candidates = {
                str(evidence.evidence_id or ""),
                str(evidence.legacy_evidence_id or ""),
            }
            if expected & {candidate for candidate in candidates if candidate}:
                citations.append(f"[证据{idx}]")
        citations = dedupe_keep_order(citations)
        if not citations:
            return answer
        return f"{answer.rstrip()} {' '.join(citations)}"
