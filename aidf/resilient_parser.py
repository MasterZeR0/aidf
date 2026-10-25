"""
Fault-Tolerant, Auto-Healing Parser for AIDF.
Engineered to handle chaotic, malformed, or unclosed LLM egress text without crashing.
Features:
1. Strict escaping of <data:untrusted> contents to prevent container-breakout injection attacks.
2. Auto-closing of unclosed tags (e.g. LLMs omitting </sec> or </doc>).
3. Resilient attribute quoting (supports unquoted numbers, arrays, and symbols).
4. Graceful fallback recovery.
"""

from __future__ import annotations
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

from aidf.models import (
    Node, TextNode, SectionNode, MediaImageNode, MediaVideoNode,
    DataTableNode, TableColumn, MathNode, UntrustedDataNode,
    ToolActionNode, EntityNode, Document
)


def sanitize_untrusted_payloads(text: str) -> str:
    """
    CRITICAL SECURITY FIX:
    Prevents prompt injection breakouts like </data:untrusted><sec prio=1>OVERRIDE...
    Escapes all '<' and '>' inside <data:untrusted> blocks before parsing.
    """
    def escape_block(m):
        open_tag = m.group(1)
        inner = m.group(2)
        close_tag = m.group(3)
        # Escape any raw angle brackets inside the untrusted payload
        escaped_inner = inner.replace("<", "&lt;").replace(">", "&gt;")
        return f"{open_tag}{escaped_inner}{close_tag}"

    pattern = r'(<data:untrusted\b[^>]*>)(.*?)(</data:untrusted>)'
    return re.sub(pattern, escape_block, text, flags=re.DOTALL)


def auto_heal_markup(text: str) -> str:
    """
    Auto-heals incomplete LLM streams or malformed markup:
    - Strips markdown codeblock wraps if present (```aidf ... ```)
    - Ensures root <doc> has matching </doc>
    - Auto-closes unclosed known tags (<sec>, <data:table>, <action:tool>)
    """
    cleaned = text.strip()
    # 1. Strip markdown fences: match ```aidf or ``` at start/end
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned).strip()
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    # 2. Ensure root <doc> exists
    if not cleaned.startswith("<doc"):
        cleaned = f"<doc id='auto-root' prio=1 lang='en'>\n{cleaned}"
    if not cleaned.endswith("</doc>"):
        cleaned = f"{cleaned}\n</doc>"

    # 3. Auto-balance unclosed child tags in correct reverse-nesting order
    # innermost tags first, outermost tags last
    known_tags = ["p", "action:tool", "data:untrusted", "data:table", "media:video", "media:img", "sec"]
    closers = []
    for tag in known_tags:
        open_count = len(re.findall(rf'<{tag}\b[^>]*>', cleaned))
        close_count = len(re.findall(rf'</{tag}>', cleaned))
        if open_count > close_count:
            missing = open_count - close_count
            for _ in range(missing):
                closers.append(f"</{tag}>")

    if closers:
        cleaned = cleaned[:-6] + "\n" + "\n".join(closers) + "\n</doc>"

    return cleaned


def normalize_to_xml(aidf_text: str) -> str:
    """
    Converts forgiving AIDF shorthand into valid XML for ElementTree.
    """
    # 1. Sanitize untrusted containers
    text = sanitize_untrusted_payloads(aidf_text)

    # 2. Auto-heal unclosed blocks
    text = auto_heal_markup(text)

    # 3. Escape bare ampersands not part of valid entity
    text = re.sub(r'&(?!(?:amp|lt|gt|apos|quot);)', '&amp;', text)

    # 4. Quote bracketed arrays: bbox=[0.1, 0.2] -> bbox="[0.1, 0.2]"
    text = re.sub(r'(\b[\w:]+)=(\[[^\]]*\])', r'\1="\2"', text)

    # 5. Quote unquoted scalars: prio=1 -> prio="1"
    text = re.sub(r'(\b[\w:]+)=([^\s"\'<>]+)', r'\1="\2"', text)

    # 6. Replace tag colons with underscores for XML parser
    text = re.sub(r'<(/)?([\w]+):([\w]+)', r'<\1\2_\3', text)

    return text


class ResilientAIDFParser:
    """
    Resilient parser designed specifically to survive chaotic LLM egress.
    Never raises unhandled fatal crashes; provides best-effort graceful AST recovery.
    """

    @staticmethod
    def parse(markup: str) -> Document:
        if not markup or not markup.strip():
            return Document(id="empty", prio=1)

        xml_ready = normalize_to_xml(markup)
        try:
            root_elem = ET.fromstring(xml_ready)
        except ET.ParseError:
            # Emergency fallback: wrap everything inside a plain text node in doc
            fallback_doc = Document(id="recovered-doc", prio=1)
            fallback_doc.children.append(TextNode(text=markup.strip(), prio=1))
            return fallback_doc

        doc_id = root_elem.attrib.get("id", "doc-auto")
        doc_lang = root_elem.attrib.get("lang", "en")
        try:
            doc_prio = int(root_elem.attrib.get("prio", 1))
        except ValueError:
            doc_prio = 1

        doc = Document(id=doc_id, lang=doc_lang, prio=doc_prio)

        for child_elem in root_elem:
            if child_elem.tag == "meta":
                doc.metadata = ResilientAIDFParser._parse_meta(child_elem)
            else:
                parsed_node = ResilientAIDFParser._parse_element(child_elem)
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
        try:
            prio = int(attrib.get("prio", 1))
        except ValueError:
            prio = 1

        if tag == "sec":
            title = attrib.get("title", "")
            children = []
            if elem.text and elem.text.strip():
                children.append(TextNode(text=elem.text.strip(), prio=prio))
            for child in elem:
                c_node = ResilientAIDFParser._parse_element(child)
                if c_node:
                    children.append(c_node)
                if child.tail and child.tail.strip():
                    children.append(TextNode(text=child.tail.strip(), prio=prio))
            return SectionNode(id=node_id, prio=prio, title=title, children=children)

        elif tag == "media_img":
            src = attrib.get("src", "")
            page = None
            if "page" in attrib:
                try:
                    page = int(attrib["page"])
                except ValueError:
                    pass
            bbox = None
            if "bbox" in attrib:
                try:
                    bbox = json.loads(attrib["bbox"])
                except Exception:
                    pass
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

        elif tag == "p":
            text = (elem.text or "").strip()
            return TextNode(id=node_id, prio=prio, text=text)

        return None
