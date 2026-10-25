"""
Pluggable Storage Engine Specification & In-Memory Reference Implementation for AIDF.
Avoids vendor lock-in or fragile 'Franken-DB' bundling by defining a clean Protocol interface
that can sit on top of In-Memory, LanceDB, DuckDB, or PostgreSQL seamlessly.
"""

from __future__ import annotations
from collections import deque
from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from aidf.models import Document, Node, SectionNode, EntityNode, DataTableNode, ReferenceNode, ParagraphNode, TextNode


class EdgeList(list):
    """List of edge dictionaries supporting both list iteration and source lookup."""
    def __getitem__(self, key):
        if isinstance(key, str):
            return [e["target"] for e in self if isinstance(e, dict) and e.get("source") == key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if isinstance(key, str):
            targets = [e["target"] for e in self if isinstance(e, dict) and e.get("source") == key]
            return targets if targets else default
        return default

    def __contains__(self, item):
        if isinstance(item, str):
            return any(e.get("source") == item or e.get("target") == item for e in self if isinstance(e, dict))
        return super().__contains__(item)


@runtime_checkable
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
        def walk(node: Node, current_doc_id: str, current_node_id: str):
            curr_id = node.id or current_node_id

            if isinstance(node, ReferenceNode):
                if node.target:
                    if current_node_id not in self._edges:
                        self._edges[current_node_id] = []
                    self._edges[current_node_id].append(node.target)
                    if node.id and node.id != current_node_id:
                        if node.id not in self._edges:
                            self._edges[node.id] = []
                        self._edges[node.id].append(node.target)

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
            if hasattr(node, "children") and isinstance(node.children, list):
                for child in node.children:
                    walk(child, current_doc_id, curr_id)

        doc_id = doc.id or "doc-default"
        walk(doc, doc_id, doc_id)

    def get_document(self, doc_id: str, max_prio: Optional[int] = None) -> Optional[Document]:
        doc = self._docs.get(doc_id)
        if not doc:
            return None
        if max_prio is not None:
            return doc.prune(max_prio)
        return doc

    def query_entities(self, entity_name: str) -> List[Dict[str, Any]]:
        return self._entity_index.get(entity_name.lower(), [])

    def query_subgraph(self, root_id: str, depth: int = 1) -> Dict[str, Any]:
        """Traverse relationships around a node using BFS."""
        visited_nodes: List[str] = [root_id]
        visited_set = {root_id}
        collected_edges = EdgeList()
        visited_edges = set()
        adj: Dict[str, List[str]] = {}

        if depth > 0:
            queue = deque([(root_id, 0)])
            while queue:
                curr, d = queue.popleft()
                if d >= depth:
                    continue
                for nbr in self._edges.get(curr, []):
                    edge_tuple = (curr, nbr)
                    if edge_tuple not in visited_edges:
                        visited_edges.add(edge_tuple)
                        collected_edges.append({"source": curr, "target": nbr})
                        if curr not in adj:
                            adj[curr] = []
                        if nbr not in adj[curr]:
                            adj[curr].append(nbr)
                    if nbr not in visited_set:
                        visited_set.add(nbr)
                        visited_nodes.append(nbr)
                        queue.append((nbr, d + 1))

        return {
            "root_id": root_id,
            "root": root_id,
            "depth": depth,
            "nodes": visited_nodes,
            "edges": collected_edges,
            "subgraph": adj,
        }
