"""Structured conspect JSON parsing."""

import json

from app.infrastructure.conspect.structure import (
    conspect_dict_to_markdown,
    try_parse_conspect_json,
)


def test_parse_minimal_json():
    raw = json.dumps(
        {
            "what_to_remember": "Rule one",
            "typical_errors": "Err",
            "algorithm": "Steps",
        },
        ensure_ascii=False,
    )
    d = try_parse_conspect_json(raw)
    assert d is not None
    assert "what_to_remember" in d


def test_to_markdown_has_headings():
    d = {
        "what_to_remember": "Body",
        "typical_errors": "E",
        "example": "Ex",
    }
    md = conspect_dict_to_markdown(d)
    assert "## Что важно запомнить" in md
    assert "Body" in md
