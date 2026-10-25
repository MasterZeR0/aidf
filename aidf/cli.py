"""
CLI tool for AIDF: Convert, Validate, Prune, and Benchmark.
"""

import sys
import argparse
from pathlib import Path

from aidf.models import Document
from aidf.parser import AIDFParser, AIDFSerializer
from aidf.converter import MarkdownToAIDFConverter


def main():
    parser = argparse.ArgumentParser(description="AIDF: Agent Ingestion Document Format CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert Markdown to AIDF
    convert_parser = subparsers.add_parser("convert", help="Convert Markdown to AIDF")
    convert_parser.add_argument("input_file", help="Input markdown file (.md)")
    convert_parser.add_argument("-o", "--output", help="Output AIDF file (.aidf)")
    convert_parser.add_argument("--id", default="doc-1", help="Document ID")

    # Prune AIDF
    prune_parser = subparsers.add_parser("prune", help="Prune AIDF by priority tier")
    prune_parser.add_argument("input_file", help="Input AIDF file (.aidf)")
    prune_parser.add_argument("--max-prio", type=int, default=1, help="Max priority to retain (1=critical, 2=normal, 3=all)")
    prune_parser.add_argument("-o", "--output", help="Output pruned AIDF file")

    # Validate AIDF
    val_parser = subparsers.add_parser("validate", help="Validate an AIDF document")
    val_parser.add_argument("input_file", help="Input AIDF file (.aidf)")

    args = parser.parse_args()

    if args.command == "convert":
        content = Path(args.input_file).read_text(encoding="utf-8")
        doc = MarkdownToAIDFConverter.convert(content, doc_id=args.id)
        serialized = AIDFSerializer.serialize(doc)
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
            print(f"✨ Converted {args.input_file} -> {args.output}")
        else:
            print(serialized)

    elif args.command == "prune":
        content = Path(args.input_file).read_text(encoding="utf-8")
        doc = AIDFParser.parse(content)
        pruned = doc.prune(max_prio=args.max_prio)
        serialized = AIDFSerializer.serialize(pruned)
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
            print(f"✨ Pruned (max_prio={args.max_prio}) -> {args.output}")
        else:
            print(serialized)

    elif args.command == "validate":
        content = Path(args.input_file).read_text(encoding="utf-8")
        try:
            doc = AIDFParser.parse(content)
            print(f"✅ Valid AIDF document: id='{doc.id}', sections/nodes={len(doc.children)}")
        except Exception as e:
            print(f"❌ AIDF Validation Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
