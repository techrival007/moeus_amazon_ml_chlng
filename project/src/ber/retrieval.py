"""Country-partitioned tantivy reference indexes and two-lane retrieval.

Each reference document carries its ordinal plus two merged term-stream
fields: one for the name lane, one for the address lane. Each merged stream
is the deduplicated union of the trigram terms, the word tokens, and the
accent-folded trigram terms of that field's views — so a single BM25 search
per lane covers typos (trigrams), short tokens (words), and Latin accent
variants (fold) at once. Each lane returns at most `lane_k` hits with
1-based ranks; compaction to reverse budgets happens in
candidates.compact_strict.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import tantivy

from ber.candidates import LaneHit
from ber.normalize import tokens, trigrams

LANE_K_DEFAULT = 8

INDEX_FIELDS = ("name_all", "addr_all")


@dataclass(frozen=True)
class IndexDoc:
    """One reference row with merged per-lane term streams, ready to index."""

    ordinal: int
    name_all: str
    addr_all: str


def merged_terms(norm: str, fold: str) -> list[str]:
    """Deduplicated union of trigrams, words, and fold-trigrams, sorted."""
    return sorted({*trigrams(norm), *tokens(norm), *trigrams(fold)})


def index_doc_from_views(
    ordinal: int, name_norm: str, addr_norm: str, name_fold: str, addr_fold: str
) -> IndexDoc:
    """Build an IndexDoc by computing merged term streams from views."""
    return IndexDoc(
        ordinal=ordinal,
        name_all=" ".join(merged_terms(name_norm, name_fold)),
        addr_all=" ".join(merged_terms(addr_norm, addr_fold)),
    )


@dataclass(frozen=True)
class QueryViews:
    """One target record's precomputed views for querying."""

    name_norm: str
    addr_norm: str
    name_fold: str
    addr_fold: str


def query_term_streams(v: QueryViews) -> dict[str, list[str]]:
    return {
        "name_all": merged_terms(v.name_norm, v.name_fold),
        "addr_all": merged_terms(v.addr_norm, v.addr_fold),
    }


def plan_lane_terms(terms: list[str], df_map: dict[str, int] | None,
                    max_terms: int) -> list[str]:
    """Select the rarest `max_terms` known terms for one lane query.

    Terms absent from the df map cannot match anything in the index and are
    dropped. Deterministic: df ascending, then term ascending.
    """
    if df_map is None or max_terms is None or len(terms) <= max_terms:
        return terms
    known = [t for t in terms if t in df_map]
    known.sort(key=lambda t: (df_map[t], t))
    return known[:max_terms]


def _schema() -> tantivy.Schema:
    b = tantivy.SchemaBuilder()
    b.add_integer_field("ref_ord", indexed=False, stored=True)
    for f in INDEX_FIELDS:
        b.add_text_field(f, stored=False)
    return b.build()


def build_index(
    index_dir: Path | str,
    docs: Iterable[IndexDoc],
    *,
    threads: int = 4,
    heap_mb: int = 512,
) -> None:
    """Build the reference index at `index_dir` (created if missing)."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(schema=_schema(), path=str(index_dir))
    writer = index.writer(heap_size=heap_mb * 1_048_576, num_threads=threads)
    for d in docs:
        doc = tantivy.Document()
        doc.add_integer("ref_ord", d.ordinal)
        for f in INDEX_FIELDS:
            doc.add_text(f, getattr(d, f))
        writer.add_document(doc)
    writer.commit()
    writer.wait_merging_threads()
    index.reload()


class Retriever:
    """Search endpoint over one built index (reopen per process/worker)."""

    def __init__(self, index_dir: Path | str):
        self.index = tantivy.Index(schema=_schema(), path=str(Path(index_dir)))
        self.index.reload()
        self.searcher = self.index.searcher()

    def _lane_search(self, field: str, terms: list[str], lane_k: int) -> list[LaneHit]:
        if not terms:
            return []
        q = self.index.parse_query("(" + " OR ".join(terms) + ")", [field])
        res = self.searcher.search(q, lane_k)
        hits: list[LaneHit] = []
        for rank, (score, addr) in enumerate(res.hits, start=1):
            ref = self.searcher.doc(addr).to_dict()["ref_ord"][0]
            hits.append(LaneHit(ref_id=int(ref), rank=rank, score=float(score)))
        return hits

    def search(
        self, q: QueryViews, *, lane_k: int = LANE_K_DEFAULT,
        df_map: dict | None = None, max_terms: int | None = None,
    ) -> tuple[list[LaneHit], list[LaneHit]]:
        """Two field lanes; inactive lanes (no terms) return no hits.

        With a df map and term cap, each lane's query keeps only its rarest
        known terms (posting-list union cost is driven by common terms).
        """
        streams = query_term_streams(q)
        name_df = df_map.get("name_all", df_map) if df_map else None
        addr_df = df_map.get("addr_all", df_map) if df_map else None
        name_hits = self._lane_search(
            "name_all", plan_lane_terms(streams["name_all"], name_df, max_terms), lane_k
        )
        addr_hits = self._lane_search(
            "addr_all", plan_lane_terms(streams["addr_all"], addr_df, max_terms), lane_k
        )
        return name_hits, addr_hits

    def search_joint(
        self, q: QueryViews, *, df_map: dict[str, dict[str, int]],
        joint_k: int = 2, max_terms: int = 8,
    ) -> list[LaneHit]:
        """Search for references sharing *both* name and address evidence.

        The conjunction avoids letting a common city or a popular business
        token alone dominate the results. This is still retrieval, before
        any pair-classifier score is computed.
        """
        if not q.name_norm or not q.addr_norm or joint_k <= 0:
            return []
        streams = query_term_streams(q)
        name = plan_lane_terms(streams["name_all"], df_map["name_all"], max_terms)
        addr = plan_lane_terms(streams["addr_all"], df_map["addr_all"], max_terms)
        if not name or not addr:
            return []
        expression = " AND ".join((
            "(" + " OR ".join(f"name_all:{term}" for term in name) + ")",
            "(" + " OR ".join(f"addr_all:{term}" for term in addr) + ")",
        ))
        query = self.index.parse_query(expression, list(INDEX_FIELDS))
        hits = self.searcher.search(query, joint_k)
        return [LaneHit(ref_id=int(self.searcher.doc(addr).to_dict()["ref_ord"][0]),
                        rank=rank, score=float(score))
                for rank, (score, addr) in enumerate(hits.hits, 1)]

    def search_many(
        self, queries: Iterable[QueryViews], *, lane_k: int = LANE_K_DEFAULT
    ) -> Iterator[tuple[list[LaneHit], list[LaneHit]]]:
        for q in queries:
            yield self.search(q, lane_k=lane_k)
