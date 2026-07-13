from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    source_type: str
    text: str
    index_text: str
    block_type: str = "text"
    page_label: int | str | None = None
    title: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source_rank: int
    source_score: float
    matched_entities: list[str] = field(default_factory=list)
    retrieval_signals: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RoutePlan:
    document: bool = True
    table: bool = True
    adela: bool = True
    reason: str = "rule-based broad retrieval"

    @property
    def sources(self) -> list[str]:
        enabled = []
        if self.document:
            enabled.append("document")
        if self.table:
            enabled.append("table")
        if self.adela:
            enabled.append("adela")
        return enabled
