"""Load SciFact dataset files into clean Python structures.

Each loader returns dicts keyed by ID for O(1) lookup during evaluation.
"""

import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document():
    """one document from the Scifact corpus"""
    doc_id: str
    title: str  
    text: str
    
    @property
    def full_text(self) -> str:
        """Title + text combined — the actual content that gets indexed/searched."""
        return f"{self.title} {self.text}"

@dataclass(frozen=True)
class Query:
      """one query from the Scifact queries"""
      query_id: str
      text: str

def load_corpus(corpus_path: Path) -> dict[str, Document]:
    """Load corpus.jsonl into a dict keyed by document ID.

    Args:
        corpus_path: Path to corpus.jsonl

    Returns:
        Mapping doc_id -> Document for O(1) lookup.
    """
    
    corpus: dict[str, Document] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            doc = Document(
                doc_id=str(raw["_id"]),
                title=raw.get("title", ""),
                text=raw.get("text", ""),
            )
            corpus[doc.doc_id] = doc
    return corpus


def load_queries(queries_path: Path) -> dict[str, Query]:
    """Load queries.jsonl into a dict keyed by query ID.

    Args:
        queries_path: Path to queries.jsonl

    Returns:
        Mapping query_id -> Query for O(1) lookup.
    """
 
    queries: dict[str, Query] = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            query = Query(query_id=str(raw["_id"]), text=raw.get("text",""))
            queries[query.query_id] = query
    return queries

import csv

def load_qrels(qrels_path: Path) -> dict[str, dict[str, int]]:
    """Load qrels/test.tsv into a dict mapping query ID to relevant doc IDs and scores.

    Args:
        qrels_path: Path to qrels TSV file (e.g., qrels/test.tsv)

    Returns:
        Mapping query_id -> {doc_id: score}.
        Only rows with score > 0 are kept (score 0 = not relevant).
    """
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    return qrels