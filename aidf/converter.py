"""
Markdown to AIDF Converter.
Converts standard markdown documents into structured AIDF.
"""

from __future__ import annotations
import re
from typing import List
from aidf.models import Document, SectionNode, TextNode, DataTableNode, TableColumn, Node


class MarkdownToAIDFConverter:
    """Converts CommonMark / GitHub-Flavored Markdown to AIDF Document AST."""

    @staticmethod
    def convert(md_text: str, doc_id: str = "doc-imported", default_prio: int = 2) -> Document:
        doc = Document(id=doc_id, prio=1)
        lines = md_text.splitlines()

        current_sec: SectionNode = None
        i = 0
        p_count = 1
        t_count = 1

        while i < len(lines):
            line = lines[i]
            trimmed = line.strip()

            # Heading (e.g. ## Section Title)
            heading_match = re.match(r'^(#{1,6})\s+(.*)', trimmed)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                prio = 1 if level <= 2 else default_prio
                sec_id = f"sec-{re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')}"
                
                new_sec = SectionNode(id=sec_id, title=title, prio=prio)
                doc.children.append(new_sec)
                current_sec = new_sec
                i += 1
                continue

            # Markdown Table detection
            if trimmed.startswith("|") and trimmed.endswith("|"):
                # Read header, separator, rows
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                
                if len(table_lines) >= 2:
                    header_cells = [c.strip() for c in table_lines[0][1:-1].split("|")]
                    columns = [TableColumn(name=c, dtype="str") for c in header_cells]
                    # skip table_lines[1] (separator |---|---|)
                    rows = []
                    for row_line in table_lines[2:]:
                        row_cells = [c.strip() for c in row_line[1:-1].split("|")]
                        rows.append(row_cells)
                    
                    tbl_node = DataTableNode(
                        id=f"tbl-{t_count}",
                        columns=columns,
                        rows=rows,
                        prio=default_prio
                    )
                    t_count += 1
                    target = current_sec.children if current_sec else doc.children
                    target.append(tbl_node)
                continue

            # Paragraph or plain text
            if trimmed:
                target = current_sec.children if current_sec else doc.children
                target.append(TextNode(id=f"p-{p_count}", text=trimmed, prio=default_prio))
                p_count += 1

            i += 1

        return doc
