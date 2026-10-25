"""
Markdown and Native Context Renderers for AIDF.
Renders an AIDF Document AST directly into clean, token-efficient CommonMark / GFM
or structured native message contexts for prompt injection.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from aidf.models import (
    Node, TextNode, ParagraphNode, ReferenceNode, SectionNode,
    MediaImageNode, MediaVideoNode, DataTableNode, MathNode,
    UntrustedDataNode, ToolActionNode, EntityNode, Document
)


class AIDFMarkdownRenderer:
    """
    Renders an AIDF Document AST into standard GitHub Flavored Markdown (GFM).
    Solves the Token Overhead problem by keeping AIDF as a rich ingestion/storage AST
    while emitting clean, highly token-dense Markdown when stuffing LLM context.
    """

    @classmethod
    def render(cls, doc: Document) -> str:
        lines: List[str] = []

        if doc.metadata:
            # Render metadata header
            src = doc.metadata.get("src")
            if src and isinstance(src, dict) and "title" in src:
                lines.append(f"# {src['title']}")
                if "uri" in src:
                    lines.append(f"*Source: {src['uri']}*")
                lines.append("")

        for child in doc.children:
            rendered = cls._render_node(child, heading_level=2)
            if rendered:
                lines.append(rendered)

        return "\n\n".join(lines).strip()

    @classmethod
    def _render_inline(cls, node: Node) -> str:
        if isinstance(node, ReferenceNode):
            # Render citation link or footnote anchor
            if node.target:
                return f"[{node.text}](#{node.target})"
            return node.text
        elif isinstance(node, EntityNode):
            return f"**{node.name}**"
        elif isinstance(node, TextNode):
            return node.text
        return getattr(node, "text", "")

    @classmethod
    def _render_node(cls, node: Node, heading_level: int = 2) -> str:
        if isinstance(node, SectionNode):
            level_str = "#" * min(heading_level, 6)
            header = f"{level_str} {node.title}"
            parts = [header]
            for child in node.children:
                child_rendered = cls._render_node(child, heading_level=heading_level + 1)
                if child_rendered:
                    parts.append(child_rendered)
            return "\n\n".join(parts)

        elif isinstance(node, ParagraphNode) or (isinstance(node, TextNode) and getattr(node, "children", None)):
            if node.children:
                inner = "".join(cls._render_inline(c) for c in node.children)
            else:
                inner = node.text
            return inner

        elif isinstance(node, TextNode):
            return node.text

        elif isinstance(node, MediaImageNode):
            caption = node.caption or "Image"
            alt = caption
            img_md = f"![{alt}]({node.src})"
            meta_details = []
            if node.page:
                meta_details.append(f"Page {node.page}")
            if node.bbox:
                meta_details.append(f"Region: {node.bbox}")
            meta_str = f"*({', '.join(meta_details)})*" if meta_details else ""

            parts = [img_md]
            if meta_str:
                parts.append(meta_str)
            if node.ocr:
                parts.append(f"**OCR Extracted:** {node.ocr}")
            if node.desc:
                parts.append(f"**Description:** {node.desc}")
            return "\n".join(parts)

        elif isinstance(node, MediaVideoNode):
            title = "Video Clip"
            timing = []
            if node.t_start:
                timing.append(f"Start: {node.t_start}")
            if node.t_end:
                timing.append(f"End: {node.t_end}")
            time_str = f" ({', '.join(timing)})" if timing else ""
            link = f"[{title}{time_str}]({node.src})"

            parts = [link]
            if node.speaker:
                parts.append(f"**Speaker:** {node.speaker}")
            if node.transcript:
                parts.append(f"> {node.transcript}")
            return "\n".join(parts)

        elif isinstance(node, DataTableNode):
            if not node.columns:
                return ""
            header_cells = [f"{col.name}" + (f" ({col.unit})" if col.unit else "") for col in node.columns]
            header_row = "| " + " | ".join(header_cells) + " |"
            sep_row = "| " + " | ".join(["---"] * len(node.columns)) + " |"
            data_rows = []
            for row in node.rows:
                # Pad or truncate row cells to match column count
                cells = list(row) + [""] * (len(node.columns) - len(row))
                cells = cells[:len(node.columns)]
                # Sanitize pipes in markdown cells
                safe_cells = [c.replace("|", "\\|") for c in cells]
                data_rows.append("| " + " | ".join(safe_cells) + " |")

            return "\n".join([header_row, sep_row] + data_rows)

        elif isinstance(node, MathNode):
            if node.mode == "display":
                return f"$$\n{node.expression}\n$$"
            return f"${node.expression}$"

        elif isinstance(node, UntrustedDataNode):
            return f"```text [UNTRUSTED DATA: source={node.source}]\n{node.content}\n```"

        elif isinstance(node, ToolActionNode):
            return f"```json [TOOL ACTION: {node.tool}]\n" + str(node.params) + "\n```"

        elif isinstance(node, EntityNode):
            return f"**{node.name}** ({node.entity_type})"

        return getattr(node, "text", "")
