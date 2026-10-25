"""
Robust Token-Aware Parser and Serializer for AIDF.
Handles concise unquoted/bracket attributes (e.g. prio=1, bbox=[...]) seamlessly.
"""

from __future__ import annotations
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from aidf.models import (
    Node, TextNode, ParagraphNode, ReferenceNode, SectionNode, MediaImageNode, MediaVideoNode,
    DataTableNode, TableColumn, MathNode, UntrustedDataNode,
    ToolActionNode, EntityNode, Document
)


def normalize_to_xml(aidf_text: str) -> str:
    """
    Normalizes compact AIDF shorthand into well-formed XML for the underlying AST parser:
    1. Escapes bare ampersands not already part of an XML entity
    2. Quotes bracketed arrays: bbox=[0.1, 0.2] -> bbox="[0.1, 0.2]"
    3. Quotes unquoted scalars: prio=1 -> prio="1"
    4. Handles colon tags: <media:img> -> <media_img>
    """
    # 1. Escape bare ampersands
    normalized = re.sub(r'&(?!(?:amp|lt|gt|apos|quot);)', '&amp;', aidf_text)

    # 2. Quote bracketed arrays: key=[...] -> key="[...]"
    def quote_bracket(m):
        attr = m.group(1)
        arr = m.group(2)
        return f'{attr}="{arr}"'

    normalized = re.sub(r'(\b[\w:]+)=(\[[^\]]*\])', quote_bracket, normalized)

    # 3. Quote unquoted scalars: key=123 or key=abc -> key="123"
    def quote_scalar(m):
        attr = m.group(1)
        val = m.group(2)
        return f'{attr}="{val}"'

    normalized = re.sub(r'(\b[\w:]+)=([^\s"\'<>]+)', quote_scalar, normalized)

    # 4. Replace tag colons with underscores for XML parser
    normalized = re.sub(r'<(/)?([\w]+):([\w]+)', r'<\1\2_\3', normalized)
    return normalized


class AIDFParser:
    """Parses AIDF markup into an AST Document."""

    @staticmethod
    def parse(markup: str) -> Document:
        xml_ready = normalize_to_xml(markup.strip())
        try:
            root_elem = ET.fromstring(xml_ready)
        except ET.ParseError as e:
            raise ValueError(f"Invalid AIDF syntax: {e}")

        if root_elem.tag != "doc":
            raise ValueError(f"Root element must be <doc>, found <{root_elem.tag}>")

        doc_id = root_elem.attrib.get("id")
        doc_lang = root_elem.attrib.get("lang", "en")
        doc_prio = int(root_elem.attrib.get("prio", 1))

        doc = Document(id=doc_id, lang=doc_lang, prio=doc_prio)

        for child_elem in root_elem:
            if child_elem.tag == "meta":
                doc.metadata = AIDFParser._parse_meta(child_elem)
            else:
                parsed_node = AIDFParser._parse_element(child_elem)
                if parsed_node:
                    doc.children.append(parsed_node)

        return doc

    @staticmethod
    def _parse_meta(elem: ET.Element) -> Dict[str, Any]:
        meta = {}
        for item in elem:
            meta[item.tag] = dict(item.attrib)
        return meta

    @staticmethod
    def _parse_element(elem: ET.Element) -> Optional[Node]:
        tag = elem.tag
        attrib = elem.attrib
        node_id = attrib.get("id")
        prio = int(attrib.get("prio", 1))

        if tag == "sec":
            title = attrib.get("title", "")
            children = []
            if elem.text and elem.text.strip():
                s_text = elem.text.strip() if len(elem) == 0 else elem.text.lstrip("\r\n")
                if s_text:
                    children.append(TextNode(text=s_text, prio=prio))
            for child in elem:
                c_node = AIDFParser._parse_element(child)
                if c_node:
                    children.append(c_node)
                if child.tail and child.tail.strip():
                    t_text = child.tail.strip() if len(elem) == 0 else child.tail.rstrip("\r\n")
                    if t_text:
                        children.append(TextNode(text=t_text, prio=prio))
            return SectionNode(id=node_id, prio=prio, title=title, children=children)

        elif tag == "media_img":
            src = attrib.get("src", "")
            page = int(attrib["page"]) if "page" in attrib else None
            bbox = None
            if "bbox" in attrib:
                try:
                    bbox = json.loads(attrib["bbox"])
                except Exception:
                    bbox = None
            caption = None
            ocr = None
            desc = None
            for sub in elem:
                if sub.tag == "caption":
                    caption = sub.text.strip() if sub.text else ""
                elif sub.tag == "ocr":
                    ocr = sub.text.strip() if sub.text else ""
                elif sub.tag == "desc":
                    desc = sub.text.strip() if sub.text else ""
            return MediaImageNode(id=node_id, prio=prio, src=src, page=page, bbox=bbox, caption=caption, ocr=ocr, desc=desc)

        elif tag == "media_video":
            src = attrib.get("src", "")
            t_start = attrib.get("t_start")
            t_end = attrib.get("t_end")
            speaker = None
            transcript = None
            for sub in elem:
                if sub.tag == "speaker":
                    speaker = sub.attrib.get("name")
                elif sub.tag == "transcript":
                    transcript = sub.text.strip() if sub.text else ""
            return MediaVideoNode(id=node_id, prio=prio, src=src, t_start=t_start, t_end=t_end, speaker=speaker, transcript=transcript)

        elif tag == "data_table":
            cols_str = attrib.get("cols", "")
            columns = []
            if cols_str:
                for col_def in cols_str.split(","):
                    col_def = col_def.strip()
                    parts = col_def.split(":")
                    name = parts[0]
                    dtype = parts[1] if len(parts) > 1 else "str"
                    unit = parts[2] if len(parts) > 2 else None
                    columns.append(TableColumn(name=name, dtype=dtype, unit=unit))

            rows = []
            raw_text = (elem.text or "").strip()
            for line in raw_text.splitlines():
                line = line.strip()
                if line.startswith("|") and line.endswith("|"):
                    cells = [c.strip() for c in line[1:-1].split("|")]
                    rows.append(cells)

            return DataTableNode(id=node_id, prio=prio, columns=columns, rows=rows)

        elif tag == "math_tex":
            mode = attrib.get("mode", "inline")
            expression = (elem.text or "").strip()
            return MathNode(id=node_id, prio=prio, expression=expression, mode=mode)

        elif tag == "data_untrusted":
            source = attrib.get("src", "unknown")
            sandbox = attrib.get("sandbox", "read-only")
            content = (elem.text or "").strip()
            return UntrustedDataNode(id=node_id, prio=prio, source=source, sandbox=sandbox, content=content)

        elif tag == "action_tool":
            tool = attrib.get("tool", "")
            method = attrib.get("method", "POST")
            params = {}
            for sub in elem:
                if sub.tag == "param":
                    key = sub.attrib.get("key")
                    if key:
                        params[key] = (sub.text or "").strip()
            return ToolActionNode(id=node_id, prio=prio, tool=tool, method=method, params=params)

        elif tag == "ent":
            name = (elem.text or "").strip()
            entity_type = attrib.get("type", "UNKNOWN")
            return EntityNode(id=node_id, prio=prio, name=name, entity_type=entity_type, metadata=dict(attrib))

        elif tag == "ref":
            target = attrib.get("target", "")
            rel = attrib.get("rel", "citation")
            text = (elem.text or "").strip()
            return ReferenceNode(id=node_id, prio=prio, target=target, rel=rel, text=text)

        elif tag == "code":
            text = elem.text or ""
            return TextNode(id=node_id, prio=prio, text=text, attrs={"tag": "code"})

        elif tag == "p":
            if len(elem) == 0:
                text = (elem.text or "").strip()
                return ParagraphNode(id=node_id, prio=prio, text=text, children=[])

            children: List[Node] = []
            text_parts: List[str] = []

            if elem.text:
                p_text = elem.text.lstrip("\r\n")
                if p_text:
                    children.append(TextNode(text=p_text, prio=prio))
                    text_parts.append(p_text)

            for child in elem:
                c_node = AIDFParser._parse_element(child)
                if c_node:
                    children.append(c_node)
                    if isinstance(c_node, ReferenceNode):
                        text_parts.append(c_node.text)
                    elif isinstance(c_node, TextNode):
                        text_parts.append(c_node.text)
                    elif isinstance(c_node, EntityNode):
                        text_parts.append(c_node.name)
                    elif hasattr(c_node, "text"):
                        text_parts.append(getattr(c_node, "text", ""))
                if child.tail:
                    t_text = child.tail.rstrip("\r\n")
                    if t_text:
                        children.append(TextNode(text=t_text, prio=prio))
                        text_parts.append(t_text)

            full_text = "".join(text_parts)
            return ParagraphNode(id=node_id, prio=prio, text=full_text, children=children)

        return None


class AIDFSerializer:
    """Serializes an AST Document back to canonical AIDF markup."""

    @staticmethod
    def serialize(doc: Document) -> str:
        lines = []
        attrs = [f'id="{doc.id}"' if doc.id else "", f'prio={doc.prio}', f'lang="{doc.lang}"']
        attr_str = " ".join([a for a in attrs if a])
        lines.append(f"<doc {attr_str}>")

        # Metadata
        if doc.metadata:
            lines.append("  <meta>")
            for k, v in doc.metadata.items():
                meta_attrs = " ".join([f'{mk}="{mv}"' for mk, mv in v.items()])
                lines.append(f"    <{k} {meta_attrs}/>")
            lines.append("  </meta>")

        for child in doc.children:
            lines.append(AIDFSerializer._serialize_node(child, depth=1))

        lines.append("</doc>")
        return "\n".join(lines)

    @staticmethod
    def _serialize_inline(node: Node) -> str:
        if isinstance(node, ReferenceNode):
            rel_str = f' rel="{node.rel}"' if node.rel and node.rel != "citation" else ""
            id_str = f' id="{node.id}"' if node.id else ""
            return f'<ref{id_str} target="{node.target}"{rel_str}>{node.text}</ref>'
        elif isinstance(node, EntityNode):
            id_str = f' id="{node.id}"' if node.id else ""
            return f'<ent{id_str} type="{node.entity_type}">{node.name}</ent>'
        elif isinstance(node, TextNode):
            return node.text
        return getattr(node, "text", "")

    @staticmethod
    def _serialize_node(node: Node, depth: int = 1) -> str:
        sp = "  " * depth
        if isinstance(node, ParagraphNode) or (isinstance(node, TextNode) and getattr(node, "children", None)):
            if node.children:
                inner = "".join(AIDFSerializer._serialize_inline(child) for child in node.children)
            else:
                inner = node.text
            if node.id:
                return f'{sp}<p id="{node.id}" prio={node.prio}>{inner}</p>'
            return f'{sp}<p prio={node.prio}>{inner}</p>'

        elif isinstance(node, TextNode):
            if node.id:
                return f'{sp}<p id="{node.id}" prio={node.prio}>{node.text}</p>'
            return f"{sp}{node.text}"

        elif isinstance(node, ReferenceNode):
            rel_str = f' rel="{node.rel}"' if node.rel and node.rel != "citation" else ""
            id_str = f' id="{node.id}"' if node.id else ""
            return f'{sp}<ref{id_str} target="{node.target}"{rel_str}>{node.text}</ref>'

        elif isinstance(node, SectionNode):
            lines = [f'{sp}<sec id="{node.id or ""}" title="{node.title}" prio={node.prio}>']
            for child in node.children:
                lines.append(AIDFSerializer._serialize_node(child, depth + 1))
            lines.append(f"{sp}</sec>")
            return "\n".join(lines)

        elif isinstance(node, MediaImageNode):
            lines = [f'{sp}<media:img id="{node.id or ""}" src="{node.src}"' + 
                     (f' page={node.page}' if node.page else '') +
                     (f' bbox={json.dumps(node.bbox)}' if node.bbox else '') +
                     f' prio={node.prio}>']
            if node.caption:
                lines.append(f"{sp}  <caption>{node.caption}</caption>")
            if node.ocr:
                lines.append(f"{sp}  <ocr>{node.ocr}</ocr>")
            if node.desc:
                lines.append(f"{sp}  <desc>{node.desc}</desc>")
            lines.append(f"{sp}</media:img>")
            return "\n".join(lines)

        elif isinstance(node, MediaVideoNode):
            lines = [f'{sp}<media:video id="{node.id or ""}" src="{node.src}"' +
                     (f' t_start="{node.t_start}"' if node.t_start else '') +
                     (f' t_end="{node.t_end}"' if node.t_end else '') +
                     f' prio={node.prio}>']
            if node.speaker:
                lines.append(f'{sp}  <speaker name="{node.speaker}"/>')
            if node.transcript:
                lines.append(f"{sp}  <transcript>{node.transcript}</transcript>")
            lines.append(f"{sp}</media:video>")
            return "\n".join(lines)

        elif isinstance(node, DataTableNode):
            cols_def = ", ".join([f"{c.name}:{c.dtype}" + (f":{c.unit}" if c.unit else "") for c in node.columns])
            lines = [f'{sp}<data:table id="{node.id or ""}" cols="{cols_def}" prio={node.prio}>']
            for row in node.rows:
                lines.append(f"{sp}  | " + " | ".join(row) + " |")
            lines.append(f"{sp}</data:table>")
            return "\n".join(lines)

        elif isinstance(node, MathNode):
            return f'{sp}<math:tex id="{node.id or ""}" mode="{node.mode}" prio={node.prio}>{node.expression}</math:tex>'

        elif isinstance(node, UntrustedDataNode):
            return f'{sp}<data:untrusted id="{node.id or ""}" src="{node.source}" sandbox="{node.sandbox}" prio={node.prio}>\n{sp}  {node.content}\n{sp}</data:untrusted>'

        elif isinstance(node, ToolActionNode):
            lines = [f'{sp}<action:tool id="{node.id or ""}" tool="{node.tool}" method="{node.method}" prio={node.prio}>']
            for k, v in node.params.items():
                lines.append(f'{sp}  <param key="{k}">{v}</param>')
            lines.append(f"{sp}</action:tool>")
            return "\n".join(lines)

        elif isinstance(node, EntityNode):
            return f'{sp}<ent id="{node.id or ""}" type="{node.entity_type}" prio={node.prio}>{node.name}</ent>'

        return f"{sp}<!-- unknown node -->"
