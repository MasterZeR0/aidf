"""
Token Benchmark: Markdown vs. AIDF (Standard) vs. AIDF (Compact) vs. Pruned
Evaluates exact token counts across realistic document scenarios.
"""

import tiktoken
from aidf.parser import AIDFParser, AIDFSerializer
from aidf.compact import CompactAIDFSerializer

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# Scenario 1: Multimodal
md_multi = """# Earnings Call Overview
Below is the quarterly revenue performance chart.

![Consolidated Income Statement](asset://page-12.png)
*(Page 12, Region: ymin=0.12, xmin=0.25, ymax=0.85, xmax=0.65)*
**OCR Text Extracted:** Total Revenues: $88,268M | Costs & Expenses: $59,743M
**Description:** Bar chart displaying quarterly growth from Q1 to Q3 2026.

### Video Segment
[Watch Call Recording (14:22 - 15:10)](asset://call-recording.mp4)
**Speaker:** Sundar Pichai
> "We continue to invest aggressively in TPUs and custom silicon infrastructure."
"""

aidf_multi = """<doc id="earnings-q3" prio=1>
  <sec id="sec-overview" title="Earnings Call Overview" prio=1>
    Below is the quarterly revenue performance chart.
    <media:img id="fig-3" src="asset://page-12.png" page=12 bbox=[0.12,0.25,0.85,0.65] prio=1>
      <caption>Consolidated Income Statement</caption>
      <ocr>Total Revenues: $88,268M | Costs & Expenses: $59,743M</ocr>
      <desc>Bar chart displaying quarterly growth from Q1 to Q3 2026.</desc>
    </media:img>
    <media:video id="vid-1" src="asset://call-recording.mp4" t_start="00:14:22" t_end="00:15:10" prio=1>
      <speaker name="Sundar Pichai"/>
      <transcript>We continue to invest aggressively in TPUs and custom silicon infrastructure.</transcript>
    </media:video>
  </sec>
</doc>"""

# Scenario 2: Tables
md_table = """## Regional Performance

| Region | Active Users | Revenue (USD M) | Growth YoY | Operating Margin | Retention Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| North America | 125,400,000 | $45,200 | +14.2% | 34.5% | 94.2% |
| EMEA | 98,200,000 | $28,150 | +11.8% | 29.1% | 91.8% |
| APAC | 165,800,000 | $31,400 | +22.4% | 24.8% | 88.5% |
| LATAM | 42,100,000 | $8,950 | +18.9% | 21.2% | 86.4% |
| Cross-regional | 12,500,000 | $3,100 | +8.5% | 18.0% | 82.0% |
"""

aidf_table = """<doc id="perf-doc" prio=1>
  <sec id="sec-regional" title="Regional Performance" prio=1>
    <data:table id="tbl-regions" cols="Region:str, Users:num, Rev:num:USD_M, YoY:pct, OpMargin:pct, Ret:pct" prio=1>
      | North America | 125400000 | 45200 | +14.2% | 34.5% | 94.2% |
      | EMEA | 98200000 | 28150 | +11.8% | 29.1% | 91.8% |
      | APAC | 165800000 | 31400 | +22.4% | 24.8% | 88.5% |
      | LATAM | 42100000 | 8950 | +18.9% | 21.2% | 86.4% |
      | Cross-regional | 12500000 | 3100 | +8.5% | 18.0% | 82.0% |
    </data:table>
  </sec>
</doc>"""

# Scenario 3: Incident Postmortem (3 Priority Tiers)
md_tiered = """# Production Incident Postmortem (INC-8821)

## Executive Summary
A database failover caused a 4-minute API degradation on checkout services.

## Detailed Timeline of Events
- 03:14 UTC: Automated health check detects elevated P99 latency on shard-4.
- 03:15 UTC: Secondary replica fails to promote due to stale lock table.
- 03:16 UTC: On-call engineer manually forces lease renewal via CLI.
- 03:17 UTC: Traffic re-balanced across pool, error rate returns to 0.00%.

## Non-Critical Appendix and Raw Metric Dumps
- Metric dump: cpu_user=42.1%, cpu_sys=12.4%, disk_io_wait=89.2ms
- Network packet drops: eth0 in=0, eth0 out=12, eth1 in=0, eth1 out=0
- Cache miss ratio during interval: redis_cluster_0=4.12%, redis_cluster_1=3.98%
- Internal thread pool state: worker_threads_busy=128, queued_tasks=14502
"""

aidf_tiered = """<doc id="inc-8821" prio=1>
  <sec id="sec-summary" title="Executive Summary" prio=1>
    <p id="p-sum" prio=1>A database failover caused a 4-minute API degradation on checkout services.</p>
  </sec>
  <sec id="sec-timeline" title="Detailed Timeline of Events" prio=2>
    <p id="p-t1" prio=2>- 03:14 UTC: Automated health check detects elevated P99 latency on shard-4.</p>
    <p id="p-t2" prio=2>- 03:15 UTC: Secondary replica fails to promote due to stale lock table.</p>
    <p id="p-t3" prio=2>- 03:16 UTC: On-call engineer manually forces lease renewal via CLI.</p>
    <p id="p-t4" prio=2>- 03:17 UTC: Traffic re-balanced across pool, error rate returns to 0.00%.</p>
  </sec>
  <sec id="sec-appendix" title="Non-Critical Appendix and Raw Metric Dumps" prio=3>
    <p id="p-m1" prio=3>- Metric dump: cpu_user=42.1%, cpu_sys=12.4%, disk_io_wait=89.2ms</p>
    <p id="p-m2" prio=3>- Network packet drops: eth0 in=0, eth0 out=12, eth1 in=0, eth1 out=0</p>
    <p id="p-m3" prio=3>- Cache miss ratio during interval: redis_cluster_0=4.12%, redis_cluster_1=3.98%</p>
    <p id="p-m4" prio=3>- Internal thread pool state: worker_threads_busy=128, queued_tasks=14502</p>
  </sec>
</doc>"""


def main():
    print("=" * 76)
    print(" 📊 AIDF vs. MARKDOWN TOKEN BENCHMARK (cl100k_base / OpenAI & Claude)")
    print("=" * 76)

    # 1. Multimodal
    doc_m = AIDFParser.parse(aidf_multi)
    compact_m = CompactAIDFSerializer.serialize(doc_m)
    print(f"\n1. MULTIMODAL INGESTION (Image + BBox + OCR + Video)")
    print(f"   • Markdown (ad-hoc prose):    {count_tokens(md_multi):>4} tokens")
    print(f"   • AIDF Standard (indented):   {count_tokens(aidf_multi):>4} tokens")
    print(f"   • AIDF Compact (prompt-mode): {count_tokens(compact_m):>4} tokens (Full AST + Bounding Boxes)")

    # 2. Structured Table
    doc_t = AIDFParser.parse(aidf_table)
    compact_t = CompactAIDFSerializer.serialize(doc_t)
    print(f"\n2. DATA TABLES (Financial Metrics)")
    print(f"   • Markdown Table (with pipes): {count_tokens(md_table):>4} tokens")
    print(f"   • AIDF Compact (typed schema): {count_tokens(compact_t):>4} tokens (Saves schema ambiguity)")

    # 3. Priority Pruning Passes
    doc_p = AIDFParser.parse(aidf_tiered)
    prio_all = CompactAIDFSerializer.serialize(doc_p)
    prio_2 = CompactAIDFSerializer.serialize(doc_p.prune(max_prio=2))
    prio_1 = CompactAIDFSerializer.serialize(doc_p.prune(max_prio=1))

    print(f"\n3. CONTEXT BUDGETING (Dynamic Priority Pruning)")
    print(f"   • Raw Markdown (Unpruned):    {count_tokens(md_tiered):>4} tokens (100.0%)")
    print(f"   • AIDF All Tiers (prio 1-3):   {count_tokens(prio_all):>4} tokens")
    print(f"   • AIDF Pruned prio<=2:         {count_tokens(prio_2):>4} tokens ({count_tokens(prio_2)/count_tokens(md_tiered)*100:.1f}% of MD, dropped raw dumps)")
    print(f"   • AIDF Pruned prio<=1 (Crit):  {count_tokens(prio_1):>4} tokens (⚡ {100 - (count_tokens(prio_1)/count_tokens(md_tiered)*100):.1f}% token reduction)")

    print("\n" + "=" * 76)

if __name__ == "__main__":
    main()
