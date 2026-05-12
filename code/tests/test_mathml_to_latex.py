"""Tests for FIPI MathML → LaTeX conversion."""

from app.infrastructure.conspect.mathml import (
    mathml_fragment_to_latex,
    problem_html_to_katex,
)


def test_msup_fraction_base():
    frag = """<m:math><m:mstyle displaystyle="true"><m:semantics><m:mrow>
<m:msup><m:mrow><m:mrow><m:mo>(</m:mo><m:mrow><m:mfrac><m:mn>1</m:mn><m:mn>7</m:mn></m:mfrac></m:mrow><m:mo>)</m:mo></m:mrow></m:mrow>
<m:mrow><m:mi>x</m:mi><m:mo>+</m:mo><m:mn>4</m:mn></m:mrow></m:msup><m:mo>=</m:mo><m:mn>49</m:mn></m:mrow>
</m:semantics></m:mstyle></m:math>"""
    tex = mathml_fragment_to_latex(frag)
    assert r"\frac{1}{7}" in tex
    assert "49" in tex


def test_problem_html_strips_tags():
    html = "<p>Найдите <m:math><m:mn>2</m:mn></m:math> число.</p>"
    out = problem_html_to_katex(html)
    assert "$2$" in out
    assert "<p>" not in out
