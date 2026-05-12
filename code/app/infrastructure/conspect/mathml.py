"""
Convert FIPI-style MathML fragments (Office XML, m: prefix) to inline LaTeX for problem_katex.
"""

from __future__ import annotations

import html as html_lib
import re
import xml.etree.ElementTree as ET


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text_content(elem: ET.Element) -> str:
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for ch in elem:
        parts.append(_elem_to_latex(ch))
        if ch.tail:
            parts.append(ch.tail)
    return "".join(parts)


def _unwrap(elem: ET.Element) -> str:
    """Process semantics / mstyle / mrow as transparent containers."""
    return _children_latex(elem)


def _children_latex(elem: ET.Element) -> str:
    parts: list[str] = []
    if elem.text:
        parts.append(_normalize_mo_text(elem.text))
    for ch in elem:
        parts.append(_elem_to_latex(ch))
        if ch.tail:
            parts.append(_normalize_mo_text(ch.tail))
    return "".join(parts)


def _normalize_mo_text(s: str) -> str:
    s = html_lib.unescape(s)
    return _mo_char(s)


def _mo_char(s: str) -> str:
    s = s.replace("\u2212", "-").replace("\u00d7", r"\cdot").replace("\u22c5", r"\cdot")
    s = s.replace("\u03c0", r"\pi")
    return s


_FUNC_NAMES = frozenset(
    {"sin", "cos", "tg", "ctg", "tan", "cot", "ln", "log", "arcsin", "arccos", "arctan"}
)


def _mi_to_latex(s: str) -> str:
    s = html_lib.unescape(s).strip()
    if not s:
        return ""
    if s in _FUNC_NAMES or (len(s) > 1 and s in _FUNC_NAMES):
        return f"\\{s}" if s in ("sin", "cos", "tan", "cot", "ln", "log") else s
    if len(s) == 1:
        return s
    return rf"\mathrm{{{s}}}"


def _wrap_brace(tex: str) -> str:
    t = tex.strip()
    if len(t) == 1 or (t.startswith("{") and t.endswith("}")):
        return t
    return "{" + t + "}"


def _elem_to_latex(elem: ET.Element) -> str:
    tag = _local(elem.tag)
    if tag in ("semantics", "mstyle", "mrow"):
        return _unwrap(elem)
    if tag == "math":
        return _unwrap(elem)
    if tag == "mi":
        return _mi_to_latex("".join(elem.itertext()))
    if tag == "mn":
        return html_lib.unescape("".join(elem.itertext())).strip()
    if tag == "mtext":
        return html_lib.unescape("".join(elem.itertext()))
    if tag == "mo":
        t = html_lib.unescape("".join(elem.itertext()))
        return _mo_char(t)
    if tag == "mspace":
        return " "
    if tag == "msup":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        base = _elem_to_latex(ch[0])
        sup = _elem_to_latex(ch[1])
        return f"{_wrap_brace(base)}^{_wrap_brace(sup)}"
    if tag == "msub":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        base = _elem_to_latex(ch[0])
        sub = _elem_to_latex(ch[1])
        return f"{_wrap_brace(base)}_{_wrap_brace(sub)}"
    if tag == "msubsup":
        ch = list(elem)
        if len(ch) < 3:
            return _children_latex(elem)
        base, sub, sup = ch[0], ch[1], ch[2]
        return f"{_wrap_brace(_elem_to_latex(base))}_{{{_elem_to_latex(sub).strip()}}}^{{{_elem_to_latex(sup).strip()}}}"
    if tag == "mfrac":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        return rf"\frac{{{_elem_to_latex(ch[0]).strip()}}}{{{_elem_to_latex(ch[1]).strip()}}}"
    if tag == "msqrt":
        inner = _children_latex(elem).strip()
        return rf"\sqrt{{{inner}}}"
    if tag == "mroot":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        rad = _elem_to_latex(ch[0]).strip()
        idx = _elem_to_latex(ch[1]).strip()
        return rf"\sqrt[{idx}]{{{rad}}}"
    if tag == "mover":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        base = _elem_to_latex(ch[0]).strip()
        accent = _elem_to_latex(ch[1]).strip()
        if accent in ("→", r"\rightarrow", r"\to"):
            return rf"\overrightarrow{{{base}}}"
        if accent in ("‾", "¯", "—") or "arrow" in (elem.get("accent") or ""):
            return rf"\overline{{{base}}}"
        return rf"\overset{{{accent}}}{{{base}}}"
    if tag == "munder":
        ch = list(elem)
        if len(ch) < 2:
            return _children_latex(elem)
        return rf"\underset{{{_elem_to_latex(ch[1]).strip()}}}{{{_elem_to_latex(ch[0]).strip()}}}"
    if tag == "mfenced":
        open_ = elem.get("open") or "("
        close = elem.get("close") or ")"
        inner = _children_latex(elem)
        if open_ == "(" and close == ")":
            return f"({inner})"
        return f"{open_}{inner}{close}"
    if tag in ("mtable", "mtr", "mtd"):
        return _children_latex(elem)
    if tag == "annotation":
        return ""
    # Unknown: flatten
    return _children_latex(elem)


_MATH_BLOCK_RE = re.compile(r"<m:math[^>]*>.*?</m:math>", re.DOTALL | re.IGNORECASE)


def mathml_fragment_to_latex(fragment: str) -> str:
    """Parse a single <m:math>...</m:math> block and return LaTeX (no $ delimiters)."""
    wrapped = f'<root xmlns:m="http://www.w3.org/1998/Math/MathML">{fragment}</root>'
    root = ET.fromstring(wrapped)
    math_el = root[0]
    raw = _elem_to_latex(math_el).strip()
    return _cleanup_latex(raw)


def _cleanup_latex(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" )", ")").replace("( ", "(")
    return s.strip()


def problem_html_to_katex(problem_html: str) -> str:
    """
    Replace each m:math block with inline $...$ and strip remaining HTML to plain text.
    """
    if not problem_html or not problem_html.strip():
        return ""

    def repl(m: re.Match[str]) -> str:
        frag = m.group(0)
        try:
            inner = mathml_fragment_to_latex(frag)
            return f" ${inner}$ "
        except ET.ParseError:
            return " "

    text = _MATH_BLOCK_RE.sub(repl, problem_html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?:\s*[.;,:]){2,}\s*$", "", text)
    return text
