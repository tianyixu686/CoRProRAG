"""Query-focused evidence packing using the deployed Qwen tokenizer."""

import math
import os
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    default = Path(__file__).resolve().parents[3] / "models/Qwen3-VL-32B-Instruct/tokenizer.json"
    path = Path(os.environ.get("COR_SIMPLE_TOKENIZER_PATH", str(default)))
    # Fail explicitly rather than silently measuring Qwen prompts with a different tokenizer.
    return Tokenizer.from_file(str(path))


def encode(text):
    return _tokenizer().encode(str(text), add_special_tokens=False).ids


def decode(tokens):
    return _tokenizer().decode(tokens, skip_special_tokens=False)


def truncate(text, budget):
    return decode(encode(text)[:max(0, budget)])


_STOP = set("a an the what which who where when how is are was were do does did of in on for to and or with from has have its it this that role play played".split())


def _words(text):
    return set(re.findall(r"[\w'-]+", text.casefold())) - _STOP


def pack_text(query, evidence, budget):
    """Select relevant whole paragraphs/table rows, retaining their source/header.

    Selection uses only the current query and retrieved sources, never gold
    answers. Original retrieval order breaks ties. Oversized paragraphs are
    split into sentences rather than discarding all text after a fixed prefix.
    """
    units = []
    for source_index, item in enumerate(evidence):
        content = str(item.get("content", ""))
        for block in re.split(r"\n\s*\n", content):
            lines = block.strip().splitlines()
            if not lines:
                continue
            row_start = next((i for i, line in enumerate(lines) if re.match(r"Row \d+:", line)), None)
            if row_start is not None:
                header = "\n".join(lines[:row_start])
                # A single table row stays attached to the document and column names.
                pieces = [header + "\n" + line for line in lines[row_start:]]
            elif len(encode(block)) > 500:
                title = lines[0] if len(lines) > 1 and len(lines[0]) < 180 else ""
                body = "\n".join(lines[1:]) if title else block
                pieces = [(title + "\n" + sentence).strip() for sentence in re.split(r"(?<=[.!?])\s+", body)]
            else:
                pieces = [block.strip()]
            for piece in pieces:
                if piece.strip():
                    units.append((source_index, piece.strip(), _words(piece)))
    if not units or budget <= 0:
        return ""
    terms = _words(query)
    weights = {term: math.log(1 + len(units) / (1 + sum(term in u[2] for u in units))) for term in terms}
    ranked = sorted(range(len(units)), key=lambda i: (-sum(weights[t] for t in terms & units[i][2]), i))
    chosen = []
    remaining = budget
    seen = set()
    for i in ranked:
        source, text, _ = units[i]
        if text in seen:
            continue
        seen.add(text)
        entry = f"[source {source + 1}]\n{text}"
        cost = len(encode(entry + "\n\n"))
        if cost <= remaining:
            chosen.append(entry)
            remaining -= cost
    # An exceptionally large single unit should not erase all evidence.
    if not chosen:
        source, text, _ = units[ranked[0]]
        return truncate(f"[source {source + 1}; excerpt]\n{text}", budget)
    return truncate("\n\n".join(chosen), budget)


def input_budget(output_tokens):
    window = int(os.environ.get("COR_SIMPLE_CONTEXT_WINDOW", "8192"))
    # Leave space for Qwen's chat template and serving-side overhead.
    return min(int(os.environ.get("COR_SIMPLE_INPUT_TOKENS", "6200")), window - output_tokens - 512)
