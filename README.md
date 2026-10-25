# AIDF (Agent Ingestion Document Format)

A multimodal, token-efficient, bracket-tagged alternative to Markdown designed for AI Agent ingestion, RAG context stuffing, and tool execution.

## Features
- **Deterministic AST Parsing**: Strict tag closures eliminate whitespace and regex parsing errors.
- **Dynamic Context Budgeting (`prio=1..3`)**: Instant token pruning before prompt injection without breaking document grammar.
- **Native Multimodal Nodes**: Spatial bounding boxes (`bbox=[ymin, xmin, ymax, xmax]`), video/audio timestamps (`t_start`, `t_end`), and parallel visual + OCR / caption tracks.
- **Typed Tabular Data**: Compact schema-defined tables (`cols="Endpoint:str, P95:num:ms"`).
- **Prompt Injection Defense**: `<data:untrusted>` sandbox containers.
- **Exact Citation Anchors**: Grounded node `#id` references.

## Quickstart

### 1. Validate an AIDF File
```bash
python3 -m aidf.cli validate document.aidf
```

### 2. Convert Markdown to AIDF
```bash
python3 -m aidf.cli convert README.md -o README.aidf
```

### 3. Prune Document for Context Windows (Retain only prio=1 critical text)
```bash
python3 -m aidf.cli prune document.aidf --max-prio 1 -o prompt_context.aidf
```

## Python API Usage
```python
from aidf.parser import AIDFParser, AIDFSerializer

raw_markup = """<doc id="report" prio=1>
  <sec id="sec-1" title="Summary" prio=1>
    <p id="p-1" prio=1>Critical launch update.</p>
    <p id="p-2" prio=3>Optional supplementary footnote.</p>
  </sec>
</doc>"""

# Parse
doc = AIDFParser.parse(raw_markup)

# Prune for prompt window (strips prio=3)
pruned_doc = doc.prune(max_prio=1)

# Serialize back to clean string
print(AIDFSerializer.serialize(pruned_doc))
```
