r"""
Tests for AIDF Strategic Fixes:
1. Handling bare '<' and mathematical comparisons without crashing.
2. Structured <data:table> with <row>[...]</row> and escaped pipes \|.
3. AIDFMarkdownRenderer for token-dense prompt rendering.
4. Robust streaming parser event handling without buffer corruption.
"""

from aidf.models import Document, SectionNode, DataTableNode, ParagraphNode
from aidf.parser import AIDFParser, AIDFSerializer
from aidf.resilient_parser import ResilientAIDFParser
from aidf.markdown_renderer import AIDFMarkdownRenderer
from aidf.streaming import AIDFStreamingParser


def test_bare_less_than_and_math():
    raw = """<doc id="math-test" prio=1>
  <sec id="sec-1" title="Logic Check" prio=1>
    <p id="p-1" prio=1>Evaluate if 5 < 10 and verify x < y in algorithm.</p>
    <p id="p-2" prio=1>Generic type is vector<string> in C++.</p>
  </sec>
</doc>"""
    # Verify standard parser does not crash on '<'
    doc = AIDFParser.parse(raw)
    assert doc.id == "math-test"
    assert len(doc.children) == 1
    sec = doc.children[0]
    assert "5 < 10" in sec.children[0].text or "5 &lt; 10" in sec.children[0].text
    print("✅ Bare '<' and comparisons parsed cleanly in AIDFParser!")


def test_table_structured_rows_and_escaped_pipes():
    # 1. JSON array row format
    raw_json_table = """<doc id="tbl-json" prio=1>
  <data:table id="tbl-1" cols="Route:str, Latency:num:ms" prio=1>
    <row>["/api/v1/search", 45]</row>
    <row>["/api/v1/cart", 120]</row>
  </data:table>
</doc>"""
    doc = AIDFParser.parse(raw_json_table)
    tbl = doc.children[0]
    assert isinstance(tbl, DataTableNode)
    assert len(tbl.rows) == 2
    assert tbl.rows[0] == ["/api/v1/search", "45"]
    assert tbl.rows[1] == ["/api/v1/cart", "120"]
    print("✅ Structured JSON row data table parsed successfully!")

    # 2. Escaped pipes in standard markdown-style table
    raw_escaped_pipe = """<doc id="tbl-pipe" prio=1>
  <data:table id="tbl-2" cols="Expression:str, Value:str" prio=1>
    | \\|x\\| | absolute value |
    | a \\| b | logical OR |
  </data:table>
</doc>"""
    doc2 = AIDFParser.parse(raw_escaped_pipe)
    tbl2 = doc2.children[0]
    assert len(tbl2.rows) == 2
    assert tbl2.rows[0][0] == "|x|"
    assert tbl2.rows[1][0] == "a | b"
    print("✅ Escaped pipes in data tables handled without breaking!")


def test_markdown_renderer():
    raw = """<doc id="render-test" prio=1>
  <meta>
    <src title="System Architecture" uri="https://internal.wiki/arch"/>
  </meta>
  <sec id="sec-overview" title="Architecture Overview" prio=1>
    <p id="p-1" prio=1>Core services communicate over gRPC.</p>
    <media:img id="fig-1" src="asset://arch.png" page=3 bbox=[0.1, 0.1, 0.8, 0.8] prio=2>
      <caption>Network Topology</caption>
      <ocr>Service Mesh 2026</ocr>
    </media:img>
    <data:table id="tbl-perf" cols="Service:str, QPS:num" prio=1>
      | auth | 15000 |
      | gateway | 50000 |
    </data:table>
  </sec>
</doc>"""
    doc = AIDFParser.parse(raw)
    md = AIDFMarkdownRenderer.render(doc)
    assert "# System Architecture" in md
    assert "## Architecture Overview" in md
    assert "Core services communicate over gRPC." in md
    assert "![Network Topology](asset://arch.png)" in md
    assert "| Service | QPS |" in md
    assert "| auth | 15000 |" in md
    print("✅ AIDF Document AST cleanly rendered to token-dense Markdown!")


def test_streaming_parser_multi_event_and_sections():
    events = []
    parser = AIDFStreamingParser(on_node_closed=lambda n: events.append(n))

    stream_chunk = (
        '<p id="p-1" prio=1>Paragraph one.</p>'
        '<p id="p-2" prio=1>Paragraph two.</p>'
        '<sec id="sec-1" title="Summary" prio=1>'
        '  <p id="p-3" prio=1>Section content.</p>'
        '</sec>'
    )
    parser.feed(stream_chunk)
    # Both paragraphs and the section should close
    assert len(events) >= 3
    print("✅ Streaming parser handles multiple closed blocks per chunk without starvation!")


if __name__ == "__main__":
    test_bare_less_than_and_math()
    test_table_structured_rows_and_escaped_pipes()
    test_markdown_renderer()
    test_streaming_parser_multi_event_and_sections()
    print("🚀 All strategic fix test suites passed successfully!")
