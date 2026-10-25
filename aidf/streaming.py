"""
Streaming Parser for AIDF (Agent Ingestion Document Format).
Allows agents to process streaming tokens and fire callbacks as complete nodes close,
preventing UI flickers and enabling immediate tool execution mid-stream.
"""

from __future__ import annotations
import re
from typing import Callable, Optional, Dict, Any, List
from aidf.models import Node, TextNode, SectionNode, DataTableNode, ToolActionNode, EntityNode
from aidf.parser import AIDFParser


class AIDFStreamEvent:
    """Event emitted during token streaming."""
    def __init__(self, event_type: str, node: Optional[Node] = None, raw_fragment: str = ""):
        self.event_type = event_type  # 'node_closed', 'tool_ready', 'fragment'
        self.node = node
        self.raw_fragment = raw_fragment

    def __repr__(self):
        return f"<AIDFStreamEvent type={self.event_type} node={type(self.node).__name__ if self.node else None}>"


class AIDFStreamingParser:
    """
    Incremental streaming tokenizer & parser for AIDF token streams.
    Fires callbacks as soon as a complete semantic block (p, sec, action:tool, data:table) closes.
    Uses index-based slicing to avoid buffer corruption and supports multiple blocks per chunk.
    """

    def __init__(self, on_node_closed: Optional[Callable[[Node], None]] = None,
                 on_tool_ready: Optional[Callable[[ToolActionNode], None]] = None):
        self.on_node_closed = on_node_closed
        self.on_tool_ready = on_tool_ready
        self.buffer = ""
        self.nodes_closed: List[Node] = []

    def feed(self, token_chunk: str):
        """Feed a new token chunk from an LLM stream."""
        self.buffer += token_chunk
        self._check_for_closed_blocks()

    def _check_for_closed_blocks(self):
        # Match complete atomic blocks inside the stream
        patterns = [
            (r'(<action:tool\b[^>]*>.*?</action:tool>)', "action_tool"),
            (r'(<data:table\b[^>]*>.*?</data:table>)', "data_table"),
            (r'(<media:img\b[^>]*>.*?</media:img>)', "media_img"),
            (r'(<media:video\b[^>]*>.*?</media:video>)', "media_video"),
            (r'(<sec\b[^>]*>.*?</sec>)', "sec"),
            (r'(<p\b[^>]*>.*?</p>)', "p"),
            (r'(<ent\b[^>]*>.*?</ent>)', "ent")
        ]

        found_any = True
        while found_any:
            found_any = False
            earliest_match = None
            earliest_span = None

            for regex_str, tag_kind in patterns:
                match = re.search(regex_str, self.buffer, re.DOTALL)
                if match:
                    span = match.span(1)
                    if earliest_span is None or span[0] < earliest_span[0]:
                        earliest_match = match.group(1)
                        earliest_span = span

            if earliest_match and earliest_span:
                raw_block = earliest_match
                start_idx, end_idx = earliest_span
                wrapped = f"<doc id='stream'>{raw_block}</doc>"
                try:
                    temp_doc = AIDFParser.parse(wrapped)
                    if temp_doc.children:
                        closed_node = temp_doc.children[0]
                        self.nodes_closed.append(closed_node)

                        if self.on_node_closed:
                            self.on_node_closed(closed_node)

                        if isinstance(closed_node, ToolActionNode) and self.on_tool_ready:
                            self.on_tool_ready(closed_node)

                        # Remove parsed block safely by exact slice index to prevent duplicate stripping
                        self.buffer = self.buffer[:start_idx] + self.buffer[end_idx:]
                        found_any = True
                except Exception:
                    # If this block failed to parse, skip past it or break
                    break
