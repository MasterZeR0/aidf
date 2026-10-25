"""
Core AST (Abstract Syntax Tree) Data Models for AIDF.
Zero heavy dependencies, highly typed, easily serializable to dict/JSON.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


@dataclass
class Node:
    """Base AST Node."""
    id: Optional[str] = None
    prio: int = 1  # 1 = critical, 2 = normal, 3 = elidable
    attrs: Dict[str, Any] = field(default_factory=dict)

    def prune(self, max_prio: int) -> Optional[Node]:
        """Return pruned node, or None if pruned out."""
        if self.prio > max_prio:
            return None
        return self


@dataclass
class TextNode(Node):
    """Raw text content."""
    text: str = ""


@dataclass
class SectionNode(Node):
    """Hierarchical section containing child nodes."""
    title: str = ""
    children: List[Node] = field(default_factory=list)

    def prune(self, max_prio: int) -> Optional[SectionNode]:
        if self.prio > max_prio:
            return None
        pruned_children = []
        for child in self.children:
            p = child.prune(max_prio)
            if p is not None:
                pruned_children.append(p)
        return SectionNode(
            id=self.id,
            prio=self.prio,
            attrs=self.attrs,
            title=self.title,
            children=pruned_children
        )


@dataclass
class MediaImageNode(Node):
    """Multimodal image node with spatial coordinates and dual text fallbacks."""
    src: str = ""
    page: Optional[int] = None
    bbox: Optional[List[float]] = None  # [ymin, xmin, ymax, xmax]
    caption: Optional[str] = None
    ocr: Optional[str] = None
    desc: Optional[str] = None


@dataclass
class MediaVideoNode(Node):
    """Multimodal video/audio segment with temporal bounds."""
    src: str = ""
    t_start: Optional[str] = None
    t_end: Optional[str] = None
    speaker: Optional[str] = None
    transcript: Optional[str] = None


@dataclass
class TableColumn:
    name: str
    dtype: str = "str"
    unit: Optional[str] = None


@dataclass
class DataTableNode(Node):
    """Structured, typed record table."""
    columns: List[TableColumn] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)


@dataclass
class MathNode(Node):
    """Mathematical expressions (LaTeX / TeX) preserved cleanly."""
    expression: str = ""
    mode: str = "inline"  # "inline" or "display"


@dataclass
class UntrustedDataNode(Node):
    """Untrusted payload container for prompt injection defense."""
    source: str = ""
    sandbox: str = "read-only"
    content: str = ""


@dataclass
class ToolActionNode(Node):
    """Directly executable tool action definition."""
    tool: str = ""
    method: str = "POST"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode(Node):
    """Knowledge graph grounded entity."""
    name: str = ""
    entity_type: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Document(Node):
    """Document root container."""
    lang: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[Node] = field(default_factory=list)

    def prune(self, max_prio: int) -> Document:
        """Returns a pruned copy of the document retaining only nodes with prio <= max_prio."""
        pruned_children = []
        for child in self.children:
            p = child.prune(max_prio)
            if p is not None:
                pruned_children.append(p)
        return Document(
            id=self.id,
            prio=self.prio,
            lang=self.lang,
            metadata=self.metadata,
            attrs=self.attrs,
            children=pruned_children
        )
