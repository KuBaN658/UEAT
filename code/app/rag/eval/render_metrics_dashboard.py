"""
Render an interactive HTML dashboard for RAG evaluation metrics.

From repo root:
    uv run python -m app.rag.eval.render_metrics_dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = EVAL_DIR / "metrics"
DEFAULT_OUT_HTML = DEFAULT_METRICS_DIR / "dashboard.html"

CONSPECT_JSON = "conspect_geval_results.json"
JUDGE_RETRIEVAL_JSON = "judge_retrieval_results.json"
BENCHMARK_JSON = "benchmark_retrieval_results.json"

METRIC_KEYS = ("personalization", "school_format", "math_correctness")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _student_sort_key(student_id: str) -> tuple[str, int, str]:
    prefix, _, suffix = student_id.rpartition("_u")
    if suffix.isdigit():
        return (prefix, int(suffix), student_id)
    return (student_id, 0, student_id)


def _json_for_script(data: dict[str, Any]) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _normalize_conspect_student(student: dict[str, Any]) -> dict[str, Any]:
    metrics = student.get("metrics") or {}
    normalized_metrics: dict[str, dict[str, Any]] = {}
    for key in METRIC_KEYS:
        metric = metrics.get(key) or {}
        normalized_metrics[key] = {
            "score": _as_float(metric.get("score")),
            "reason": str(metric.get("reason") or ""),
            "success": bool(metric.get("success", False)),
            "error": bool(metric.get("error", False)),
        }

    return {
        "student_id": str(student.get("student_id") or ""),
        "overall_score": _as_float(student.get("overall_score")),
        "metrics": normalized_metrics,
        "has_error": bool(student.get("has_error", False)),
    }


def _normalize_retrieval_student(student: dict[str, Any]) -> dict[str, Any]:
    results = []
    for rank, item in enumerate(student.get("results") or [], start=1):
        results.append(
            {
                "rank": rank,
                "atom_id": str(item.get("atom_id") or ""),
                "atom_title": str(item.get("atom_title") or ""),
                "retrieval_score": _as_float(item.get("retrieval_score")),
                "judge_score": _as_float(item.get("judge_score")),
                "is_relevant": bool(item.get("is_relevant", False)),
                "matched_task_numbers": item.get("matched_task_numbers") or [],
                "rationale": str(item.get("rationale") or ""),
                "raw_response": str(item.get("raw_response") or ""),
            }
        )

    retrieved_count = _as_int(student.get("retrieved_count"))
    relevant_count = _as_int(student.get("relevant_count"))
    relevant_ratio = relevant_count / retrieved_count if retrieved_count else None

    return {
        "student_id": str(student.get("student_id") or ""),
        "wrong_tasks_count": _as_int(student.get("wrong_tasks_count")),
        "queries": [str(query) for query in student.get("queries") or []],
        "avg_judge_score": _as_float(student.get("avg_judge_score")),
        "relevant_count": relevant_count,
        "retrieved_count": retrieved_count,
        "relevant_ratio": relevant_ratio,
        "results": results,
    }


def _merge_students(
    conspect_students: list[dict[str, Any]],
    retrieval_students: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}

    for student in conspect_students:
        student_id = student["student_id"]
        if not student_id:
            continue
        by_id.setdefault(student_id, {"student_id": student_id})["conspect"] = student

    for student in retrieval_students:
        student_id = student["student_id"]
        if not student_id:
            continue
        by_id.setdefault(student_id, {"student_id": student_id})["retrieval"] = student

    merged = []
    for student_id in sorted(by_id, key=_student_sort_key):
        item = by_id[student_id]
        item.setdefault("conspect", None)
        item.setdefault("retrieval", None)
        merged.append(item)
    return merged


def _build_summary(
    geval: dict[str, Any],
    judge: dict[str, Any],
    benchmark: dict[str, Any],
    metrics_dir: Path,
) -> dict[str, Any]:
    retrieval_students = judge.get("students") or []
    total_relevant = sum(_as_int(student.get("relevant_count")) for student in retrieval_students)
    total_retrieved = sum(_as_int(student.get("retrieved_count")) for student in retrieval_students)
    relevant_ratio = total_relevant / total_retrieved if total_retrieved else None

    return {
        "dashboard_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_dir": str(metrics_dir),
        "sources": {
            CONSPECT_JSON: bool(geval),
            JUDGE_RETRIEVAL_JSON: bool(judge),
            BENCHMARK_JSON: bool(benchmark),
        },
        "conspect": {
            "generated_at_utc": geval.get("_generated_at_utc"),
            "student_count": _as_int(geval.get("student_count")),
            "excluded_count": _as_int(geval.get("excluded_count")),
            "overall_avg_score": _as_float(geval.get("overall_avg_score")),
            "per_metric_avg": {
                key: _as_float((geval.get("per_metric_avg") or {}).get(key)) for key in METRIC_KEYS
            },
        },
        "retrieval_judge": {
            "generated_at_utc": judge.get("_generated_at_utc"),
            "student_count": _as_int(judge.get("student_count")),
            "overall_avg_judge_score": _as_float(judge.get("overall_avg_judge_score")),
            "total_relevant": total_relevant,
            "total_retrieved": total_retrieved,
            "relevant_ratio": relevant_ratio,
        },
        "benchmark": {
            "generated_at_utc": benchmark.get("_generated_at_utc"),
            "evaluated": _as_int((benchmark.get("all_tasks") or {}).get("evaluated")),
            "k_values": benchmark.get("k_values") or [],
            "task_numbers": benchmark.get("task_numbers") or [],
        },
    }


def build_dashboard_data(metrics_dir: Path) -> dict[str, Any]:
    conspect_payload = _read_json(metrics_dir / CONSPECT_JSON)
    judge_payload = _read_json(metrics_dir / JUDGE_RETRIEVAL_JSON)
    benchmark_payload = _read_json(metrics_dir / BENCHMARK_JSON)

    geval = dict(conspect_payload.get("geval") or {})
    judge = dict(judge_payload.get("judge") or {})
    benchmark = dict(benchmark_payload.get("benchmark") or {})

    geval["_generated_at_utc"] = conspect_payload.get("generated_at_utc")
    judge["_generated_at_utc"] = judge_payload.get("generated_at_utc")
    benchmark["_generated_at_utc"] = benchmark_payload.get("generated_at_utc")

    conspect_students = [
        _normalize_conspect_student(student) for student in geval.get("students") or []
    ]
    retrieval_students = [
        _normalize_retrieval_student(student) for student in judge.get("students") or []
    ]

    return {
        "summary": _build_summary(geval, judge, benchmark, metrics_dir),
        "students": _merge_students(conspect_students, retrieval_students),
        "benchmark": benchmark,
    }


def render_dashboard_html(data: dict[str, Any]) -> str:
    return HTML_TEMPLATE.replace("__DASHBOARD_DATA__", _json_for_script(data))


def write_dashboard(metrics_dir: Path, out_path: Path) -> None:
    data = build_dashboard_data(metrics_dir)
    html = render_dashboard_html(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render interactive metrics dashboard HTML.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=f"Directory with metrics JSON files (default: {DEFAULT_METRICS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_HTML,
        help=f"Output HTML path (default: {DEFAULT_OUT_HTML})",
    )
    args = parser.parse_args(argv)

    if not args.metrics_dir.exists():
        print(f"Metrics directory does not exist: {args.metrics_dir}", file=sys.stderr)
        return 1

    write_dashboard(metrics_dir=args.metrics_dir, out_path=args.out)
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Metrics Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #374151;
      --accent: #38bdf8;
      --good: #34d399;
      --warn: #fbbf24;
      --bad: #f87171;
      --shadow: rgba(0, 0, 0, 0.24);
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f8fafc;
        --panel: #ffffff;
        --panel-soft: #f1f5f9;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e2e8f0;
        --shadow: rgba(15, 23, 42, 0.08);
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    header {
      padding: 28px 32px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), transparent 46%);
    }
    h1, h2, h3 { margin: 0; }
    h1 { font-size: clamp(26px, 4vw, 42px); letter-spacing: -0.04em; }
    h2 { font-size: 20px; margin-bottom: 14px; }
    h3 { font-size: 16px; margin-bottom: 10px; }
    .subtle { color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: minmax(0, 35%) minmax(0, 65%);
      gap: 18px;
      padding: 22px 32px 36px;
    }
    .full { grid-column: 1 / -1; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 16px 40px var(--shadow);
      padding: 18px;
      min-width: 0;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(160px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel-soft);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .value {
      font-size: 27px;
      font-weight: 750;
      margin-top: 6px;
      letter-spacing: -0.03em;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .chart { width: 100%; height: 390px; }
    .chart-small { height: 320px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    input, select, button {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 9px 11px;
      background: var(--panel-soft);
      color: var(--text);
      font: inherit;
    }
    button {
      cursor: pointer;
      transition: border-color 0.15s, transform 0.15s;
    }
    button:hover { border-color: var(--accent); transform: translateY(-1px); }
    button.active {
      background: rgba(56, 189, 248, 0.16);
      border-color: var(--accent);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      overflow: hidden;
      border-radius: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      color: var(--muted);
      background: var(--panel-soft);
      position: sticky;
      top: 0;
      z-index: 1;
      cursor: pointer;
      user-select: none;
    }
    tr.clickable { cursor: pointer; }
    tr.clickable:hover, tr.selected { background: rgba(56, 189, 248, 0.1); }
    .score {
      display: inline-flex;
      min-width: 56px;
      justify-content: center;
      border-radius: 999px;
      padding: 3px 8px;
      font-variant-numeric: tabular-nums;
      background: var(--panel-soft);
    }
    .score.good { color: var(--good); }
    .score.warn { color: var(--warn); }
    .score.bad { color: var(--bad); }
    .details-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }
    .metric-detail {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: var(--panel-soft);
    }
    .reason {
      max-height: 160px;
      overflow: auto;
      color: var(--muted);
      font-size: 13px;
      white-space: pre-wrap;
    }
    .query-list {
      display: grid;
      gap: 8px;
      margin: 12px 0;
    }
    .query-button {
      text-align: left;
      width: 100%;
    }
    .notice {
      border: 1px solid rgba(251, 191, 36, 0.45);
      background: rgba(251, 191, 36, 0.1);
      border-radius: 14px;
      padding: 10px 12px;
      color: var(--muted);
      margin: 10px 0;
      font-size: 14px;
    }
    .atoms {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .atom {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: var(--panel-soft);
    }
    .atom-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .atom-title { font-weight: 700; }
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }
    .badge {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
    }
    pre {
      overflow: auto;
      white-space: pre-wrap;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      font-size: 12px;
    }
    @media (max-width: 1180px) {
      main { grid-template-columns: 1fr; padding: 18px; }
      header { padding: 24px 18px 14px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .details-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>RAG Metrics Dashboard</h1>
    <p class="subtle" id="generatedAt"></p>
  </header>

  <main>
    <section class="panel full">
      <div class="cards" id="summaryCards"></div>
    </section>

    <section class="panel full">
      <h2>IR Benchmark</h2>
      <div id="benchmarkChart" class="chart chart-small"></div>
    </section>

    <section class="panel full">
      <h2>Общий обзор</h2>
      <div class="grid">
        <div>
          <h3>Score по студентам / конспектам</h3>
          <div id="overviewChart" class="chart"></div>
        </div>
        <div>
          <h3>G-Eval метрики</h3>
          <div id="metricChart" class="chart"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="toolbar">
        <div>
          <h2>Студенты / конспекты</h2>
          <div class="subtle">Клик по строке или Plotly-графику открывает детализацию.</div>
        </div>
        <input id="studentSearch" type="search" placeholder="Фильтр по student_id">
      </div>
      <div style="overflow:auto; max-height: 1360px;">
        <table>
          <thead>
            <tr>
              <th data-sort="student_id">student_id</th>
              <th data-sort="conspect">G-Eval</th>
              <th data-sort="retrieval">Retrieval judge</th>
              <th data-sort="relevant">Relevant</th>
            </tr>
          </thead>
          <tbody id="studentsTable"></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Детализация</h2>
      <div id="studentDetails"></div>
      <h3 style="margin-top: 18px;">Рассеяние извлеченных атомов</h3>
      <div id="atomScatter" class="chart chart-small"></div>
    </section>
  </main>

  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const METRIC_KEYS = ["personalization", "school_format", "math_correctness"];
    const METRIC_LABELS = {
      personalization: "Personalization",
      school_format: "School format",
      math_correctness: "Math correctness",
    };
    const IR_METRICS = ["precision", "recall", "ndcg", "mrr", "map", "hitrate"];
    const IR_LABELS = {
      precision: "Precision",
      recall: "Recall",
      ndcg: "NDCG",
      mrr: "MRR",
      map: "MAP",
      hitrate: "HitRate",
    };

    let selectedStudentId = DATA.students[0]?.student_id || null;
    let selectedQueryIndex = 0;
    let sortState = { key: "student_id", direction: 1 };

    const plotConfig = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    };

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function num(value) {
      return typeof value === "number" && Number.isFinite(value) ? value : null;
    }

    function formatScore(value) {
      const n = num(value);
      return n === null ? "—" : n.toFixed(3);
    }

    function formatPercent(value) {
      const n = num(value);
      return n === null ? "—" : `${(n * 100).toFixed(1)}%`;
    }

    function scoreClass(value) {
      const n = num(value);
      if (n === null) return "";
      if (n >= 0.8) return "good";
      if (n >= 0.5) return "warn";
      return "bad";
    }

    function scorePill(value) {
      return `<span class="score ${scoreClass(value)}">${formatScore(value)}</span>`;
    }

    function truncate(value, max = 130) {
      const text = String(value ?? "");
      return text.length > max ? `${text.slice(0, max - 1)}…` : text;
    }

    function getStudent(studentId) {
      return DATA.students.find((student) => student.student_id === studentId) || null;
    }

    function filteredStudents() {
      const query = document.getElementById("studentSearch").value.trim().toLowerCase();
      const rows = DATA.students.filter((student) => {
        return !query || student.student_id.toLowerCase().includes(query);
      });
      return rows.sort((a, b) => {
        const av = sortValue(a, sortState.key);
        const bv = sortValue(b, sortState.key);
        if (av < bv) return -1 * sortState.direction;
        if (av > bv) return 1 * sortState.direction;
        return a.student_id.localeCompare(b.student_id, "ru", { numeric: true });
      });
    }

    function sortValue(student, key) {
      if (key === "conspect") return num(student.conspect?.overall_score) ?? -1;
      if (key === "retrieval") return num(student.retrieval?.avg_judge_score) ?? -1;
      if (key === "relevant") return num(student.retrieval?.relevant_ratio) ?? -1;
      return student.student_id;
    }

    function plotLayout(title = "") {
      const styles = getComputedStyle(document.documentElement);
      return {
        title,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: styles.getPropertyValue("--text").trim() },
        margin: { l: 48, r: 22, t: title ? 42 : 18, b: 72 },
        hovermode: "closest",
        legend: { orientation: "h", y: 1.12 },
        xaxis: { gridcolor: styles.getPropertyValue("--border").trim(), automargin: true },
        yaxis: {
          range: [0, 1.05],
          gridcolor: styles.getPropertyValue("--border").trim(),
          zerolinecolor: styles.getPropertyValue("--border").trim(),
        },
      };
    }

    function renderSummaryCards() {
      const summary = DATA.summary;
      const cards = [
        {
          label: "G-Eval overall",
          value: formatScore(summary.conspect.overall_avg_score),
          hint: `${summary.conspect.student_count} конспектов`,
        },
        {
          label: "Personalization",
          value: formatScore(summary.conspect.per_metric_avg.personalization),
          hint: "avg score",
        },
        {
          label: "School format",
          value: formatScore(summary.conspect.per_metric_avg.school_format),
          hint: "avg score",
        },
        {
          label: "Math correctness",
          value: formatScore(summary.conspect.per_metric_avg.math_correctness),
          hint: "avg score",
        },
        {
          label: "Retrieval judge",
          value: formatScore(summary.retrieval_judge.overall_avg_judge_score),
          hint: `${formatPercent(summary.retrieval_judge.relevant_ratio)} relevant`,
        },
      ];
      document.getElementById("summaryCards").innerHTML = cards
        .map((card) => `
          <div class="card">
            <div class="label">${esc(card.label)}</div>
            <div class="value">${esc(card.value)}</div>
            <div class="subtle">${esc(card.hint)}</div>
          </div>
        `)
        .join("");
      document.getElementById("generatedAt").textContent =
        `Dashboard generated: ${summary.dashboard_generated_at_utc}`;
    }

    function renderOverviewChart() {
      const students = filteredStudents();
      const ids = students.map((student) => student.student_id);
      const conspectScores = students.map((student) => num(student.conspect?.overall_score));
      const retrievalScores = students.map((student) => num(student.retrieval?.avg_judge_score));

      const traces = [
        {
          type: "bar",
          name: "G-Eval overall",
          x: ids,
          y: conspectScores,
          customdata: ids,
          marker: { color: "#38bdf8" },
          hovertemplate: "%{x}<br>G-Eval=%{y:.3f}<extra></extra>",
        },
        {
          type: "bar",
          name: "Retrieval judge avg",
          x: ids,
          y: retrievalScores,
          customdata: ids,
          marker: { color: "#34d399" },
          hovertemplate: "%{x}<br>Retrieval judge=%{y:.3f}<extra></extra>",
        },
      ];
      const layout = plotLayout();
      layout.barmode = "group";
      Plotly.newPlot("overviewChart", traces, layout, plotConfig).then((chart) => {
        chart.on("plotly_click", (event) => selectStudent(event.points[0]?.customdata));
      });
    }

    function renderMetricChart() {
      const students = filteredStudents().filter((student) => student.conspect);
      const ids = students.map((student) => student.student_id);
      const traces = METRIC_KEYS.map((key, idx) => ({
        type: "bar",
        name: METRIC_LABELS[key],
        x: ids,
        y: students.map((student) => num(student.conspect.metrics[key]?.score)),
        customdata: ids,
        marker: { color: ["#38bdf8", "#a78bfa", "#fbbf24"][idx] },
        hovertemplate: `%{x}<br>${METRIC_LABELS[key]}=%{y:.3f}<extra></extra>`,
      }));
      const layout = plotLayout();
      layout.barmode = "group";
      Plotly.newPlot("metricChart", traces, layout, plotConfig).then((chart) => {
        chart.on("plotly_click", (event) => selectStudent(event.points[0]?.customdata));
      });
    }

    function renderBenchmarkChart() {
      const allTasks = DATA.benchmark.all_tasks || {};
      const macro = allTasks.macro_averages || {};
      const kValues = (DATA.benchmark.k_values || []).map(String);
      const traces = kValues.map((k) => ({
        type: "bar",
        name: `@${k}`,
        x: IR_METRICS.map((metric) => IR_LABELS[metric]),
        y: IR_METRICS.map((metric) => num(macro[k]?.[metric])),
        customdata: IR_METRICS.map((metric) => `@${k} ${IR_LABELS[metric]}`),
        hovertemplate: "%{customdata}<br>score=%{y:.3f}<extra></extra>",
      }));
      const layout = plotLayout();
      layout.barmode = "group";
      layout.xaxis.title = "metric";
      layout.yaxis.title = "macro average";
      Plotly.newPlot("benchmarkChart", traces, layout, plotConfig);
    }

    function renderAtomScatter() {
      const student = getStudent(selectedStudentId);
      const results = student?.retrieval?.results || [];
      const colors = results.map((item) => (item.is_relevant ? "#34d399" : "#f87171"));
      const trace = {
        type: "scatter",
        mode: "markers",
        x: results.map((item) => num(item.retrieval_score)),
        y: results.map((item) => num(item.judge_score)),
        text: results.map((item) => `${item.rank}. ${item.atom_title}`),
        customdata: results.map((item) => item.atom_id),
        marker: { color: colors, size: 11, line: { width: 1, color: "rgba(255,255,255,0.45)" } },
        hovertemplate:
          "%{text}<br>retrieval=%{x:.3f}<br>judge=%{y:.3f}<br>%{customdata}<extra></extra>",
      };
      const layout = plotLayout();
      layout.xaxis.title = "retrieval_score";
      layout.yaxis.title = "judge_score";
      Plotly.newPlot("atomScatter", [trace], layout, plotConfig);
    }

    function renderStudentsTable() {
      const rows = filteredStudents();
      document.getElementById("studentsTable").innerHTML = rows
        .map((student) => {
          const retrieval = student.retrieval;
          const relevantText = retrieval
            ? `${retrieval.relevant_count}/${retrieval.retrieved_count} (${formatPercent(retrieval.relevant_ratio)})`
            : "—";
          return `
            <tr class="clickable ${student.student_id === selectedStudentId ? "selected" : ""}"
                data-student-id="${esc(student.student_id)}">
              <td><strong>${esc(student.student_id)}</strong></td>
              <td>${scorePill(student.conspect?.overall_score)}</td>
              <td>${scorePill(retrieval?.avg_judge_score)}</td>
              <td>${esc(relevantText)}</td>
            </tr>
          `;
        })
        .join("");

      document.querySelectorAll("#studentsTable tr").forEach((row) => {
        row.addEventListener("click", () => selectStudent(row.dataset.studentId));
      });
    }

    function renderStudentDetails() {
      const student = getStudent(selectedStudentId);
      const container = document.getElementById("studentDetails");
      if (!student) {
        container.innerHTML = '<p class="subtle">Нет выбранного студента.</p>';
        return;
      }

      const conspect = student.conspect;
      const retrieval = student.retrieval;
      const queryButtons = (retrieval?.queries || [])
        .map((query, index) => `
          <button class="query-button ${index === selectedQueryIndex ? "active" : ""}"
                  data-query-index="${index}">
            <strong>Query ${index + 1}</strong><br>${esc(truncate(query, 220))}
          </button>
        `)
        .join("");

      const selectedQuery = retrieval?.queries?.[selectedQueryIndex] || "";
      const metricDetails = METRIC_KEYS.map((key) => {
        const metric = conspect?.metrics?.[key];
        return `
          <div class="metric-detail">
            <div class="label">${esc(METRIC_LABELS[key])}</div>
            <div class="value">${formatScore(metric?.score)}</div>
            <div class="reason">${esc(metric?.reason || "Нет данных.")}</div>
          </div>
        `;
      }).join("");

      const atomCards = (retrieval?.results || [])
        .slice()
        .sort((a, b) => (num(b.judge_score) ?? -1) - (num(a.judge_score) ?? -1))
        .map((item) => `
          <article class="atom">
            <div class="atom-head">
              <div>
                <div class="atom-title">#${item.rank} ${esc(item.atom_title)}</div>
                <div class="subtle">${esc(item.atom_id)}</div>
              </div>
              ${scorePill(item.judge_score)}
            </div>
            <div class="badges">
              <span class="badge">retrieval=${formatScore(item.retrieval_score)}</span>
              <span class="badge">${item.is_relevant ? "relevant" : "not relevant"}</span>
              <span class="badge">tasks: ${esc((item.matched_task_numbers || []).join(", ") || "—")}</span>
            </div>
            <div class="reason">${esc(item.rationale || "Нет rationale.")}</div>
            ${item.raw_response ? `
              <details>
                <summary>raw_response</summary>
                <pre>${esc(item.raw_response)}</pre>
              </details>
            ` : ""}
          </article>
        `)
        .join("");

      container.innerHTML = `
        <h3>${esc(student.student_id)}</h3>
        <div class="badges">
          <span class="badge">G-Eval: ${formatScore(conspect?.overall_score)}</span>
          <span class="badge">Retrieval judge: ${formatScore(retrieval?.avg_judge_score)}</span>
          <span class="badge">Relevant: ${retrieval ? `${retrieval.relevant_count}/${retrieval.retrieved_count}` : "—"}</span>
        </div>

        <div class="details-grid">${metricDetails}</div>

        <h3>Запросы</h3>
        <div class="query-list">${queryButtons || '<p class="subtle">Запросов нет.</p>'}</div>
        ${selectedQuery ? `
          <div class="notice">
            <strong>Выбранный запрос:</strong><br>${esc(selectedQuery)}
            <br><br>
            Текущий JSON хранит retrieved-атомы общим списком для студента, без отдельной выдачи по каждому запросу.
          </div>
        ` : ""}

        <h3>Извлеченные атомы</h3>
        <div class="atoms">${atomCards || '<p class="subtle">Извлеченные атомы отсутствуют.</p>'}</div>
      `;

      document.querySelectorAll(".query-button").forEach((button) => {
        button.addEventListener("click", () => {
          selectedQueryIndex = Number(button.dataset.queryIndex || 0);
          renderStudentDetails();
        });
      });
    }

    function selectStudent(studentId) {
      if (!studentId || !getStudent(studentId)) return;
      selectedStudentId = studentId;
      selectedQueryIndex = 0;
      renderStudentsTable();
      renderStudentDetails();
      renderAtomScatter();
    }

    function renderAll() {
      renderSummaryCards();
      renderOverviewChart();
      renderMetricChart();
      renderBenchmarkChart();
      renderStudentsTable();
      renderStudentDetails();
      renderAtomScatter();
    }

    document.getElementById("studentSearch").addEventListener("input", () => {
      renderOverviewChart();
      renderMetricChart();
      renderStudentsTable();
    });

    document.querySelectorAll("th[data-sort]").forEach((header) => {
      header.addEventListener("click", () => {
        const key = header.dataset.sort;
        if (sortState.key === key) {
          sortState.direction *= -1;
        } else {
          sortState = { key, direction: key === "student_id" ? 1 : -1 };
        }
        renderOverviewChart();
        renderMetricChart();
        renderStudentsTable();
      });
    });

    renderAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
