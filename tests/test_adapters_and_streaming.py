"""
Tests for AIDF Streaming Parser and RAG Framework Adapters.
"""

from aidf.streaming import AIDFStreamingParser
from aidf.adapters import AIDFLangChainLoader, AIDFLlamaIndexReader
from aidf.models import ToolActionNode, TextNode

def test_streaming():
    print("Testing AIDF incremental streaming parser...")
    events = []
    tools_triggered = []

    def on_node(n):
        events.append(n)

    def on_tool(t):
        tools_triggered.append(t)

    parser = AIDFStreamingParser(on_node_closed=on_node, on_tool_ready=on_tool)

    # Simulated LLM token stream
    token_stream = [
        '<doc id="stream-test" prio=1>\n',
        '  <sec id="sec-1" title="Streamed Section" prio=1>\n',
        '    <p id="p-1" prio=1>First streamed ',
        'sentence completed.</p>\n',
        '    <action:tool id="act-1" tool="execute_trade" method="POST" prio=1>\n',
        '      <param key="symbol">GOOG</param>\n',
        '      <param key="qty">100</param>\n',
        '    </action:tool>\n',
        '  </sec>\n',
        '</doc>'
    ]

    for chunk in token_stream:
        parser.feed(chunk)

    assert len(events) >= 2, f"Expected at least 2 events, got {len(events)}"
    assert len(tools_triggered) == 1, "Expected 1 tool triggered mid-stream"
    assert tools_triggered[0].tool == "execute_trade"
    assert tools_triggered[0].params["symbol"] == "GOOG"
    print("✅ Streaming parser test passed (Tool triggered directly mid-stream)!")


def test_adapters():
    print("Testing LangChain and LlamaIndex adapters on sample.aidf...")
    loader = AIDFLangChainLoader(file_path="/Users/nehaverma/aidf/sample.aidf", max_prio=2)
    docs = loader.load()
    assert len(docs) >= 1
    assert "doc_id" in docs[0].metadata
    print(f"✅ LangChain loader produced {len(docs)} document chunks with metadata: {docs[0].metadata}")

    reader = AIDFLlamaIndexReader(max_prio=2)
    nodes = reader.load_data("/Users/nehaverma/aidf/sample.aidf")
    assert len(nodes) >= 1
    print(f"✅ LlamaIndex reader produced {len(nodes)} node chunks: {nodes[0].id_}")


if __name__ == "__main__":
    test_streaming()
    test_adapters()
