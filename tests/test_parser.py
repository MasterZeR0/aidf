from aidf.models import Document, SectionNode, MediaImageNode, DataTableNode, TableColumn, TextNode
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

if __name__ == "__main__":
    test_full_pipeline()
