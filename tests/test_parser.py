from aidf.models import (
    Document, SectionNode, MediaImageNode, DataTableNode, TableColumn,
    TextNode, ParagraphNode, ReferenceNode
)
from aidf.parser import AIDFParser, AIDFSerializer

def test_full_pipeline():
    sample_aidf = """<doc id="sample-q3" prio=1 lang="en">
  <meta>
    <src title="Quarterly Report" uri="s3://reports/q3.pdf"/>
  </meta>
  <sec id="sec-overview" title="Financial Highlights" prio=1>
    <p id="p-1" prio=1>Company revenue reached record levels in Q3.</p>
    <data:table id="tbl-rev" cols="Quarter:str, Revenue:num:USD_M, Growth:pct" prio=1>
      | Q3 2025 | 76693 | +11% |
      | Q3 2026 | 88268 | +15% |
    </data:table>
    <sec id="sec-subsegment" title="Elidable details" prio=3>
      <p id="p-detail" prio=3>Minor warehouse maintenance occurred in July.</p>
    </sec>
  </sec>
  <media:img id="fig-margins" src="asset://margins.png" page=14 bbox=[0.1, 0.2, 0.8, 0.9] prio=2>
    <caption>Operating Margins 2026</caption>
    <ocr>Operating Margin: 32%</ocr>
  </media:img>
</doc>"""

    # 1. Parse
    doc = AIDFParser.parse(sample_aidf)
    assert doc.id == "sample-q3"
    assert len(doc.children) == 2
    
    sec = doc.children[0]
    assert isinstance(sec, SectionNode)
    assert sec.title == "Financial Highlights"
    
    table = sec.children[1]
    assert isinstance(table, DataTableNode)
    assert len(table.columns) == 3
    assert table.columns[1].unit == "USD_M"
    assert len(table.rows) == 2

    # 2. Priority Pruning Pass (e.g. prune prio=3 to fit tight context)
    pruned_doc = doc.prune(max_prio=2)
    pruned_sec = pruned_doc.children[0]
    # Verify prio=3 section was pruned
    assert len(pruned_sec.children) == 2  # p-1 and tbl-rev remain, sec-subsegment was dropped

    # 3. Serialize back
    serialized = AIDFSerializer.serialize(pruned_doc)
    assert "Financial Highlights" in serialized
    assert "Minor warehouse maintenance" not in serialized
    assert "Operating Margin: 32%" in serialized  # prio=2 remains

    print("✅ All AIDF AST, parsing, pruning, and serialization tests passed!")


def test_mixed_content_paragraph():
    markup = '<doc id="test-mixed"><p>Cloud grew <ref target="tbl-1">15%</ref> this quarter.</p></doc>'
    doc = AIDFParser.parse(markup)
    assert doc.id == "test-mixed"
    assert len(doc.children) == 1

    p = doc.children[0]
    assert isinstance(p, (ParagraphNode, TextNode))
    assert len(p.children) == 3

    # Check child nodes and preserved text
    assert isinstance(p.children[0], TextNode)
    assert p.children[0].text == "Cloud grew "

    assert isinstance(p.children[1], ReferenceNode)
    assert p.children[1].target == "tbl-1"
    assert p.children[1].text == "15%"
    assert p.children[1].rel == "citation"

    assert isinstance(p.children[2], TextNode)
    assert p.children[2].text == " this quarter."

    # Full text preservation without truncation
    assert p.text == "Cloud grew 15% this quarter."

    # Serialization roundtrip preserves reference tag and text
    serialized = AIDFSerializer.serialize(doc)
    assert '<ref target="tbl-1">15%</ref>' in serialized
    assert "Cloud grew " in serialized
    assert " this quarter." in serialized
    print("✅ Mixed content paragraph parsing & reference node test passed!")


if __name__ == "__main__":
    test_full_pipeline()
    test_mixed_content_paragraph()
