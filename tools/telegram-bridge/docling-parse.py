#!/usr/bin/env python3

import json
import sys
from importlib import metadata
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: docling-parse.py <input-path>")

    input_path = Path(sys.argv[1]).expanduser().resolve()
    if not input_path.exists():
        fail(f"input file not found: {input_path}")

    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
    except Exception as error:  # pragma: no cover - exercised via bridge error handling
        fail(
            "Docling is not available in the configured Python environment. "
            "Install docling and its chunking dependencies on the bridge host. "
            f"Import error: {error}"
        )

    try:
        converter = DocumentConverter()
        document = converter.convert(str(input_path)).document
        parsed_markdown = document.export_to_markdown()
        chunker = HybridChunker()
        chunks = []
        for index, chunk in enumerate(chunker.chunk(dl_doc=document)):
            contextual_text = (chunker.contextualize(chunk=chunk) or "").strip()
            if not contextual_text:
                continue
            source_text = getattr(chunk, "text", "") or ""
            chunks.append({
                "chunk_index": index,
                "source_text": str(source_text).strip(),
                "contextual_text": contextual_text,
            })
    except Exception as error:  # pragma: no cover - exercised via bridge error handling
        fail(f"Docling failed to parse {input_path.name}: {error}")

    output = {
        "title": input_path.name,
        "parsed_markdown": parsed_markdown,
        "plain_text": parsed_markdown,
        "docling_version": metadata.version("docling"),
        "chunks": chunks,
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
