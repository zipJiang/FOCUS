"""Conservative parsing of structured compaction notes, outside reasoning only."""

from __future__ import annotations

import re
from dataclasses import dataclass

SUMMARY_INSTRUCTION = """
Reply with exactly one <summary>...</summary> block, outside any thinking.
Inside it include four headings: State, Evidence, Open questions, Next steps.
Each section must have content (write None if empty). Record established facts only;
do not verify them again. End immediately after </summary>.
"""
REPAIR_INSTRUCTION = """
Your previous compaction was unusable. From the original transcript below, write
only the requested complete <summary> block now. Keep reasoning minimal.
"""


@dataclass(frozen=True)
class ParsedSummary:
    notes: str
    recovered: bool = False


def parse_summary(raw: str, *, prefix: str = "", at_cap: bool = False) -> ParsedSummary:
    """Recover wrappers only when all four populated sections are unambiguous.

    Missing closing wrappers at the output cap are truncation, not safe recovery.
    A fully closed block remains usable even if subsequent output was truncated.
    """
    if prefix.rstrip().endswith("<think>"):
        if "</think>" not in raw:
            raise ValueError("unclosed_reasoning")
        raw = raw.split("</think>", 1)[1]
    # Reject unmatched reasoning; never promote notes quoted inside thinking.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "<think>" in raw:
        raw = raw.split("<think>", 1)[0]
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    blocks = re.findall(r"<summary\s*>(.*?)</summary\s*>", raw, re.S | re.I)
    if len(blocks) > 1:
        raise ValueError("ambiguous_summary")
    recovered = not blocks
    if blocks:
        notes = blocks[0].strip()
    else:
        if at_cap:
            raise ValueError("truncated_summary")
        notes = re.sub(r"</?summary\s*>", "", raw, flags=re.I).strip()
    notes = re.sub(r"^```[^\n]*\n|\n```$", "", notes).strip()
    heading = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
        r"(State|Evidence|Open[ _-]+questions|Next[ _-]+steps)"
        r"(?:\*\*)?\s*:?(?:\*\*)?\s*$",
        re.I | re.M,
    )
    matches = list(heading.finditer(notes))
    names = [re.sub(r"[ _-]+", " ", m[1].lower()) for m in matches]
    if names != ["state", "evidence", "open questions", "next steps"]:
        raise ValueError("missing_or_ambiguous_sections")
    if notes[: matches[0].start()].strip():
        raise ValueError("commentary_before_notes")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(notes)
        if not notes[match.end() : end].strip():
            raise ValueError("empty_section")
    if re.search(r"</?(?:summary|think)\b", notes, re.I):
        raise ValueError("nested_markup")
    return ParsedSummary(notes, recovered)
