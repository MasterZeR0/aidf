"""
Compact Token Serializer for AIDF.
Eliminates gratuitous indentation spaces and empty closures when serializing for LLM prompt ingestion.
"""

from __future__ import annotations
import json
from aidf.models import (
    Node, TextNode, SectionNode, MediaImageNode, MediaVideoNode,
    DataTableNode, MathNode, UntrustedDataNode, ToolActionNode,
    EntityNode, Document
)


class CompactAIDFSerializer:
    """Serializes AIDF into minimal-token format for prompt injection."""

    @staticmethod
    def serialize(doc: Document) -> str:
        attrs = [f'id="{doc.id}"' if doc.id else "", f'prio={doc.prio}', f'lang="{doc.lang}"']
        attr_str = " ".join([a for a in attrs if a])
        out = [f"<doc {attr_str}>"]

        for child in doc.children:
            node_str = CompactAIDFSerializer._serialize_node(child)
            if node_str:
                out.append(node_str)

        out.append("</doc>")
        return "\n".join(out)

    @staticmethod
    def _serialize_node(node: Node) -> str:
        if isinstance(node, TextNode):
            if node.id:
                return f'<p id="{node.id}" prio={node.prio}>{node.text}</p>'
            return node.text

        elif isinstance(node, SectionNode):
            lines = [f'<sec id="{node.id or ""}" title="{node.title}" prio={node.prio}>']
            for child in node.children:
                s = CompactAIDFSerializer._serialize_node(child)
                if s:
                    lines.append(s)
            lines.append("</sec>")
            return "\n".join(lines)

        elif isinstance(node, MediaImageNode):
            lines = [f'<media:img id="{node.id or ""}" src="{node.src}"' + 
                     (f' page={node.page}' if node.page else '') +
                     (f' bbox={json.dumps(node.bbox)}' if node.bbox else '') +
                     f' prio={node.prio}>']
            if node.caption:
                lines.append(f"<caption>{node.caption}</caption>")
            if node.ocr:
                lines.append(f"<ocr>{node.ocr}</ocr>")
            if node.desc:
                lines.append(f"<desc>{node.desc}</desc>")
            lines.append("</media:img>")
            return "\n".join(lines)

        elif isinstance(node, MediaVideoNode):
            lines = [f'<media:video id="{node.id or ""}" src="{node.src}"' +
                     (f' t_start="{node.t_start}"' if node.t_start else '') +
                     (f' t_end="{node.t_end}"' if node.t_end else '') +
                     f' prio={node.prio}>']
            if node.speaker:
                lines.append(f'<speaker name="{node.speaker}"/>')
            if node.transcript:
                lines.append(f"<transcript>{node.transcript}</transcript>")
            lines.append("</media:video>")
            return "\n".join(lines)

        elif isinstance(node, DataTableNode):
            cols_def = ",".join([f"{c.name}:{c.dtype}" + (f":{c.unit}" if c.unit else "") for c in node.columns])
            lines = [f'<data:table id="{node.id or ""}" cols="{cols_def}" prio={node.prio}>']
            for row in node.rows:
                lines.append("|" + "|".join(row) + "|")
            lines.append("</data:table>")
            return "\n".join(lines)

        elif isinstance(node, MathNode):
            return f'<math:tex id="{node.id or ""}" mode="{node.mode}" prio={node.prio}>{node.expression}</math:tex>'

        elif isinstance(node, UntrustedDataNode):
            return f'<data:untrusted id="{node.id or ""}" src="{node.source}" sandbox="{node.sandbox}" prio={node.prio}>{node.content}</data:untrusted>'

        elif isinstance(node, ToolActionNode):
            lines = [f'<action:tool id="{node.id or ""}" tool="{node.tool}" method="{node.method}" prio={node.prio}>']
            for k, v in node.params.items():
                lines.append(f'<param key="{k}">{v}</param>')
            lines.append("</action:tool>")
            return "\n".join(lines)

        elif isinstance(node, EntityNode):
            return f'<ent id="{node.id or ""}" type="{node.entity_type}" prio={node.prio}>{node.name}</ent>'

        return ""
