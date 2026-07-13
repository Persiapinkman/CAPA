from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException

from ace_rag.ace.service import AceService
from ace_rag.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PlaybookOrganizeRequest,
    PlaybookOrganizeResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from ace_rag.core.config import get_settings
from ace_rag.playbook.store import PlaybookStore
from ace_rag.v2_client.client import V2Client, V2ClientError


router = APIRouter(prefix="/playbook", tags=["ACE Playbook"])


@lru_cache()
def get_store() -> PlaybookStore:
    return PlaybookStore(get_settings().PLAYBOOK_DB_PATH)


@lru_cache()
def get_v2_client() -> V2Client:
    settings = get_settings()
    return V2Client(settings.V2_BASE_URL, timeout_seconds=settings.V2_TIMEOUT_SECONDS)


@lru_cache()
def get_ace_service() -> AceService:
    settings = get_settings()
    return AceService(
        store=get_store(),
        v2_client=get_v2_client(),
        seed_path=settings.PLAYBOOK_SEED_PATH,
        auto_import_seed=settings.AUTO_IMPORT_SEED,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    store = get_store()
    v2_payload: dict[str, Any]
    status = "ok"
    started = time.perf_counter()
    try:
        v2_payload = await get_v2_client().health()
        v2_payload["reachable"] = True
    except Exception as exc:
        status = "degraded"
        v2_payload = {"reachable": False, "error": str(exc)}
    v2_payload["health_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return HealthResponse(
        status=status,
        service="ace-rag",
        v2=v2_payload,
        playbook={
            "db_path": str(get_settings().PLAYBOOK_DB_PATH),
            "active_items": store.get_item_count(),
        },
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    try:
        return await get_ace_service().retrieve(request)
    except V2ClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if request.stream:
        raise HTTPException(status_code=400, detail="ace-rag does not support streaming in this version")
    try:
        return await get_ace_service().query(request)
    except V2ClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    return await get_ace_service().add_feedback(request)


@router.post("/organize", response_model=PlaybookOrganizeResponse)
async def organize(request: PlaybookOrganizeRequest) -> PlaybookOrganizeResponse:
    return get_ace_service().organize_playbook(request)
