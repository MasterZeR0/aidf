"""
Pluggable Storage Engine Specification & In-Memory Reference Implementation for AIDF.
Avoids vendor lock-in or fragile 'Franken-DB' bundling by defining a clean Protocol interface
that can sit on top of In-Memory, LanceDB, DuckDB, or PostgreSQL seamlessly.
"""

from __future__ import annotations
from typing import Protocol, List, Dict, Any, Optional
from aidf.models import Document, Node, SectionNode, EntityNode, DataTableNode


class AIDFStorageBackend(Protocol):
    """Abstract interface that any DB (DuckDB, LanceDB, Postgres, In-Memory) can implement."""

    def put_document(self, doc: Document) -> str:
        """Store or update an AIDF document."""
        ...

    def get_document(self, doc_id: str, max_prio: Optional[int] = None) -> Optional[Document]:
        """Retrieve a document with optional priority pruning."""
        ...

    def query_entities(self, entity_name: str) -> List[EntityNode]:
        """Find entity nodes across documents."""
        ...

    def query_subgraph(self, root_id: str, depth: int = 1) -> Dict[str, Any]:
        """Traverse relationships around a node."""
        ...


class InMemoryAIDFStore:
    """
    Zero-dependency, clean reference storage engine.
    Indexes documents, builds entity inverted indices, and enables graph traversal.
    """

    def __init__(self):
        self._docs: Dict[str, Document] = {}
        self._entity_index: Dict[str, List[Dict[str, Any]]] = {}
        self._edges: Dict[str, List[str]] = {}  # source_id -> [target_ids]

    def put_document(self, doc: Document) -> str:
        doc_id = doc.id or "doc-default"
        self._docs[doc_id] = doc
        self._index_doc(doc)
        return doc_id

    def _index_doc(self, doc: Document):
        def walk(node: Node, current_doc_id: str):
            if isinstance(node, EntityNode):
                key = node.name.lower()
                if key not in self._entity_index:
                    self._entity_index[key] = []
                self._entity_index[key].append({
                    "doc_id": current_doc_id,
                    "node_id": node.id,
                    "type": node.entity_type
                })
            
            # Recurse children
            if isinstance(node, (Document, SectionNode)):
                for child in node.children:
                    walk(child, current_doc_id)

        walk(doc, doc.id or "doc-default")

    def get_document(self, doc_id: str, max_prio: Optional[int] = None) -> Optional[Document]:
        doc = self._docs.get(doc_id)
        if not doc:
            return None
        if max_prio is not None:
            return doc.prune(max_prio)
        return doc

    def query_entities(self, entity_name: str) -> List[Dict[str, Any]]:
        return self._entity_index.get(entity_name.lower(), [])
