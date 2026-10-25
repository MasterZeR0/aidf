"""
LangChain and LlamaIndex Document Adapters for AIDF.
Loads AIDF documents into framework-native Document and Node abstractions with metadata & chunking.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path

from aidf.models import Document, SectionNode, TextNode, DataTableNode, MediaImageNode
from aidf.parser import AIDFParser


# -----------------------------------------------------------------------------
# LangChain Adapter
# -----------------------------------------------------------------------------
class AIDFLangChainLoader:
    """
    Loads AIDF documents into standard LangChain Document objects.
    Preserves section hierarchy, priority, and element ID in metadata.
    """

    def __init__(self, file_path: str, max_prio: int = 2):
        self.file_path = file_path
        self.max_prio = max_prio

    def load(self) -> List[Any]:
        # Deferred import so langchain is not a strict hard dependency
        try:
            from langchain_core.documents import Document as LCDocument
        except ImportError:
            # Fallback stub matching LangChain Document signature
            class LCDocument:
                def __init__(self, page_content: str, metadata: Dict[str, Any]):
                    self.page_content = page_content
                    self.metadata = metadata
                def __repr__(self):
                    return f"Document(page_content='{self.page_content[:30]}...', metadata={self.metadata})"

        text = Path(self.file_path).read_text(encoding="utf-8")
        aidf_doc = AIDFParser.parse(text)
        pruned_doc = aidf_doc.prune(max_prio=self.max_prio)

        lc_docs = []
        for child in pruned_doc.children:
            if isinstance(child, SectionNode):
                # Flatten section content
                content_parts = []
                for sub in child.children:
                    if isinstance(sub, TextNode):
                        content_parts.append(sub.text)
                    elif isinstance(sub, DataTableNode):
                        content_parts.append(f"[Table: {len(sub.rows)} rows]")
                
                lc_docs.append(LCDocument(
                    page_content="\n".join(content_parts),
                    metadata={
                        "doc_id": pruned_doc.id,
                        "section_id": child.id,
                        "title": child.title,
                        "prio": child.prio,
                        "source": self.file_path
                    }
                ))
            elif isinstance(child, TextNode):
                lc_docs.append(LCDocument(
                    page_content=child.text,
                    metadata={"doc_id": pruned_doc.id, "node_id": child.id, "prio": child.prio}
                ))

        return lc_docs


# -----------------------------------------------------------------------------
# LlamaIndex Adapter
# -----------------------------------------------------------------------------
class AIDFLlamaIndexReader:
    """
    Reads AIDF documents into LlamaIndex BaseNode/Document structures.
    """

    def __init__(self, max_prio: int = 2):
        self.max_prio = max_prio

    def load_data(self, file_path: str) -> List[Any]:
        try:
            from llama_index.core.schema import TextNode as LINode
        except ImportError:
            class LINode:
                def __init__(self, text: str, id_: str, metadata: Dict[str, Any]):
                    self.text = text
                    self.id_ = id_
                    self.metadata = metadata
                def __repr__(self):
                    return f"TextNode(id='{self.id_}', text='{self.text[:30]}...', metadata={self.metadata})"

        text = Path(file_path).read_text(encoding="utf-8")
        aidf_doc = AIDFParser.parse(text).prune(max_prio=self.max_prio)

        nodes = []
        for child in aidf_doc.children:
            if isinstance(child, SectionNode):
                sec_text = "\n".join([c.text for c in child.children if isinstance(c, TextNode)])
                nodes.append(LINode(
                    text=sec_text,
                    id_=child.id or "sec",
                    metadata={"title": child.title, "prio": child.prio, "doc_id": aidf_doc.id}
                ))
        return nodes
