"""Tests for conspect format validation."""

from app.infrastructure.conspect.quality import scan_conspect_violations


def test_scan_finds_missing_practice_section():
    text = "## Что важно\n\nx"
    v = scan_conspect_violations(text)
    assert any("missing_practice" in x for x in v)


def test_scan_valid_minimal_structure():
    text = """## Потренируйся

1. Задание №6: foo
2. Задание №10: bar
3. Задание №12: baz

## Чеклист
"""
    v = scan_conspect_violations(text)
    assert not v


def test_forbidden_derivative_in_practice():
    text = """## Потренируйся

1. Задание №6: ok
2. Найдите производную функции
3. Задание №10: ok
4. Задание №12: ok

## Чеклист
"""
    v = scan_conspect_violations(text)
    assert any("производную" in x.lower() for x in v)
