#!/usr/bin/env python3
"""Visualize unified benchmark outputs with SVG charts and HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "N/A"


def _fmt_rate(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "N/A"


def _clip_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    text = text.replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _bar_chart_svg(
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    width: int = 960,
    height: int = 440,
    y_as_percent: bool = False,
) -> str:
    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_value = max(values) if values else 0.0
    if max_value <= 0:
        max_value = 1.0
    y_max = max_value * 1.15
    if y_as_percent and y_max < 1.0:
        y_max = 1.0

    n = max(len(values), 1)
    gap = 14
    bar_width = max((plot_width - gap * (n + 1)) / n, 10)

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" fill="#111827">{_escape(title)}</text>')

    x0 = margin_left
    y0 = margin_top + plot_height
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#374151" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#374151" stroke-width="1.5"/>')

    for tick in range(6):
        ratio = tick / 5.0
        y = y0 - ratio * plot_height
        value = ratio * y_max
        label = f"{value * 100:.0f}%" if y_as_percent else f"{value:.0f}"
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#4b5563">{_escape(label)}</text>')

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = x0 + gap + idx * (bar_width + gap)
        h = (value / y_max) * plot_height
        y = y0 - h
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" '
            'fill="#2563eb" opacity="0.9" rx="4" />'
        )
        value_text = f"{value * 100:.1f}%" if y_as_percent else f"{value:.1f}"
        svg.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 6, margin_top + 12):.2f}" '
            f'text-anchor="middle" font-size="12" fill="#111827">{_escape(value_text)}</text>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y0 + 20:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#111827">{_escape(label)}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _histogram_svg(
    title: str,
    values: Sequence[float],
    bins: int = 12,
    width: int = 960,
    height: int = 440,
) -> str:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        cleaned = [0.0]

    low = min(cleaned)
    high = max(cleaned)
    if math.isclose(low, high):
        high = low + 1.0
    step = (high - low) / max(bins, 1)
    counts = [0] * bins
    for value in cleaned:
        idx = int((value - low) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    margin_left = 80
    margin_right = 30
    margin_top = 55
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1
    if max_count <= 0:
        max_count = 1

    gap = 6
    bar_width = max((plot_width - gap * (bins + 1)) / bins, 6)
    x0 = margin_left
    y0 = margin_top + plot_height

    svg: List[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" fill="#111827">{_escape(title)}</text>')
    svg.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#374151" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{margin_left + plot_width}" y2="{y0}" stroke="#374151" stroke-width="1.5"/>')

    for tick in range(6):
        ratio = tick / 5.0
        y = y0 - ratio * plot_height
        count_value = int(round(ratio * max_count))
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#4b5563">{count_value}</text>')

    for idx, count in enumerate(counts):
        x = x0 + gap + idx * (bar_width + gap)
        h = (count / max_count) * plot_height
        y = y0 - h
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" '
            'fill="#059669" opacity="0.85" rx="3" />'
        )

        if idx % 2 == 0:
            bucket_low = low + idx * step
            svg.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y0 + 20:.2f}" text-anchor="middle" '
                f'font-size="10" fill="#111827">{bucket_low:.0f}</text>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_html_report(
    summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
    chart_dir_name: str,
    top_n: int,
) -> str:
    counts = summary.get("counts", {})
    retrieval = summary.get("retrieval_metrics", {})
    answer = summary.get("answer_metrics", {})
    latency = summary.get("latency_ms", {})
    source_breakdown = summary.get("source_type_breakdown", {})
    meta = summary.get("meta", {})

    slowest_rows = sorted(
        rows,
        key=lambda row: float(row.get("client_total_ms") or 0.0),
        reverse=True,
    )[:top_n]
    bad_rows = [
        row
        for row in rows
        if (not row.get("success"))
        or (float(row.get("evidence_recall_at_k", 0.0)) < 1.0)
        or (not row.get("answer_correct"))
    ][:top_n]

    source_lines: List[str] = []
    for source_type, payload in source_breakdown.items():
        source_lines.append(
            "<tr>"
            f"<td>{_escape(source_type)}</td>"
            f"<td>{_escape(payload.get('count', 0))}</td>"
            f"<td>{_fmt_rate(payload.get('fused_source_hit_rate', 0.0))}</td>"
            f"<td>{_fmt_rate(payload.get('evidence_recall_at_k', 0.0))}</td>"
            f"<td>{_fmt_rate(payload.get('answer_correct_rate', 0.0))}</td>"
            f"<td>{_fmt_rate(payload.get('avg_char_f1', 0.0))}</td>"
            "</tr>"
        )
    source_table = (
        "\n".join(source_lines)
        if source_lines
        else "<tr><td colspan='6'>N/A</td></tr>"
    )

    slow_rows_html: List[str] = []
    for row in slowest_rows:
        slow_rows_html.append(
            "<tr>"
            f"<td>{_escape(row.get('benchmark_id'))}</td>"
            f"<td>{_escape(row.get('run_index'))}</td>"
            f"<td>{_fmt_ms(row.get('client_total_ms'))}</td>"
            f"<td>{_fmt_ms(row.get('retrieve_ms'))}</td>"
            f"<td>{_fmt_ms(row.get('answer_ms'))}</td>"
            f"<td>{_escape(_clip_text(row.get('query', ''), 180))}</td>"
            "</tr>"
        )
    slow_table = (
        "\n".join(slow_rows_html)
        if slow_rows_html
        else "<tr><td colspan='6'>N/A</td></tr>"
    )

    bad_rows_html: List[str] = []
    for row in bad_rows:
        reference_answer = _clip_text(row.get("reference_answer", ""), 360)
        answer_text = row.get("answer_text") or row.get("answer_preview") or ""
        answer_text = _clip_text(answer_text, 560)
        keyword_ratio = f"{row.get('keyword_hit', 0)}/{row.get('keyword_total', 0)}"
        compare_cell = (
            "<details>"
            "<summary>展开对比</summary>"
            "<div class='compare'>"
            f"<div><b>参考答案</b><br/>{_escape(reference_answer).replace(chr(10), '<br/>')}</div>"
            f"<div><b>模型答案</b><br/>{_escape(answer_text).replace(chr(10), '<br/>')}</div>"
            "</div>"
            "</details>"
        )

        bad_rows_html.append(
            "<tr>"
            f"<td>{_escape(row.get('benchmark_id'))}</td>"
            f"<td>{_escape(row.get('run_index'))}</td>"
            f"<td>{_escape(row.get('success'))}</td>"
            f"<td>{_fmt_rate(row.get('evidence_recall_at_k', 0.0))}</td>"
            f"<td>{_escape(row.get('evidence_first_hit_rank'))}</td>"
            f"<td>{_escape(keyword_ratio)}</td>"
            f"<td>{_fmt_rate(row.get('answer_score', 0.0))}</td>"
            f"<td>{_escape(row.get('exact_match'))}</td>"
            f"<td>{_escape(row.get('reference_containment'))}</td>"
            f"<td>{_escape(_clip_text(row.get('query', ''), 120))}</td>"
            f"<td>{compare_cell}</td>"
            f"<td>{_escape(_clip_text(row.get('error_message') or '', 120))}</td>"
            "</tr>"
        )

    bad_table = (
        "\n".join(bad_rows_html)
        if bad_rows_html
        else "<tr><td colspan='12'>N/A</td></tr>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Unified Benchmark Report</title>
  <style>
    :root {{
      --card: #ffffff;
      --text: #1f2937;
      --muted: #4b5563;
      --line: #d1d5db;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, #e2f3ff 0%, #f6f8fb 40%, #f9fafb 100%);
      line-height: 1.5;
    }}
    .wrap {{
      max-width: 1320px;
      margin: 24px auto 48px;
      padding: 0 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 16px;
      box-shadow: 0 6px 20px rgba(17, 24, 39, 0.05);
    }}
    h1, h2 {{
      margin: 0 0 10px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; color: #0f172a; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }}
    .kpi {{
      background: linear-gradient(135deg, #eff6ff, #ecfeff);
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 12px; }}
    .kpi .value {{
      margin-top: 4px;
      font-size: 20px;
      font-weight: 700;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f3f4f6; font-weight: 600; }}
    .chart {{
      width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fff;
      margin-bottom: 10px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .def-box {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: #1e293b;
      margin-bottom: 8px;
    }}
    details summary {{
      cursor: pointer;
      color: #0f766e;
      font-weight: 600;
    }}
    .compare {{
      margin-top: 6px;
      white-space: pre-wrap;
      line-height: 1.45;
      max-width: 580px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Unified Benchmark Report</h1>
      <div class="meta">
        Generated at: {_escape(datetime.now().isoformat(timespec='seconds'))}<br/>
        Benchmark: {_escape(meta.get('benchmark_path', 'N/A'))}<br/>
        API: {_escape(meta.get('api_url', 'N/A'))}<br/>
        Output Dir: {_escape(meta.get('output_dir', 'N/A'))}
      </div>
    </div>

    <div class="card">
      <h2>Headline Metrics</h2>
      <div class="grid">
        <div class="kpi"><div class="label">Total Requests</div><div class="value">{_escape(counts.get('total_requests', 0))}</div></div>
        <div class="kpi"><div class="label">Success Rate</div><div class="value">{_fmt_rate(counts.get('success_rate', 0.0))}</div></div>
        <div class="kpi"><div class="label">Evidence Recall@K</div><div class="value">{_fmt_rate(retrieval.get('avg_evidence_recall_at_k', 0.0))}</div></div>
        <div class="kpi"><div class="label">Answer Correct Rate</div><div class="value">{_fmt_rate(answer.get('answer_correct_rate', 0.0))}</div></div>
        <div class="kpi"><div class="label">Latency P95 (Client)</div><div class="value">{_fmt_ms((latency.get('client_total_ms') or {{}}).get('p95'))} ms</div></div>
        <div class="kpi"><div class="label">Avg Char F1</div><div class="value">{_fmt_rate(answer.get('avg_char_f1', 0.0))}</div></div>
      </div>
    </div>

    <div class="card">
      <h2>Metric Definitions</h2>
      <div class="def-box"><b>exact_match</b>：参考答案与模型答案在去空白、去标点、统一大小写后，文本完全一致。</div>
      <div class="def-box"><b>reference_containment</b>：规范化后，参考答案是模型答案的连续子串，或模型答案是参考答案的连续子串。</div>
      <div class="def-box">两者都接近 0 的常见情况：模型回答更自由（复述、扩展、解释）或关键字段有偏差，导致连续字符串匹配失败。</div>
    </div>

    <div class="card">
      <h2>Charts</h2>
      <img class="chart" src="{_escape(chart_dir_name)}/latency_avg.svg" alt="Latency Avg" />
      <img class="chart" src="{_escape(chart_dir_name)}/latency_p95.svg" alt="Latency P95" />
      <img class="chart" src="{_escape(chart_dir_name)}/latency_hist.svg" alt="Latency Histogram" />
      <img class="chart" src="{_escape(chart_dir_name)}/retrieval_answer_metrics.svg" alt="Retrieval and Answer Metrics" />
      <img class="chart" src="{_escape(chart_dir_name)}/source_breakdown.svg" alt="Source Breakdown" />
    </div>

    <div class="card">
      <h2>Source-Type Breakdown</h2>
      <table>
        <thead>
          <tr>
            <th>Source</th><th>Count</th><th>Fused Source Hit</th><th>Evidence Recall@K</th><th>Answer Correct</th><th>Avg Char F1</th>
          </tr>
        </thead>
        <tbody>{source_table}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Top Slow Queries</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Run</th><th>Client ms</th><th>Retrieve ms</th><th>Answer ms</th><th>Query</th>
          </tr>
        </thead>
        <tbody>{slow_table}</tbody>
      </table>
    </div>

    <div class="card">
      <h2>Risk Cases (with Answer Comparison)</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Run</th><th>Success</th><th>Evidence Recall</th><th>First Hit Rank</th><th>KW Hit</th><th>Answer Score</th><th>Exact Match</th><th>Containment</th><th>Query</th><th>Reference vs Answer</th><th>Error</th>
          </tr>
        </thead>
        <tbody>{bad_table}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG charts and HTML report from unified benchmark outputs."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="directory generated by unified_benchmark_eval.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="report output directory (default: <result-dir>/viz)",
    )
    parser.add_argument("--top-n", type=int, default=20, help="rows shown in slow/risk tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result-dir not found: {result_dir}")

    summary = _load_json(result_dir / "summary.json")
    rows = _load_jsonl(result_dir / "requests.jsonl")

    output_dir = Path(args.output_dir) if args.output_dir else (result_dir / "viz")
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    latency = summary.get("latency_ms", {})
    retrieval = summary.get("retrieval_metrics", {})
    answer = summary.get("answer_metrics", {})
    source_breakdown = summary.get("source_type_breakdown", {})

    latency_avg_labels = ["client", "server", "route", "retrieve", "fuse", "answer"]
    latency_avg_values = [
        float((latency.get("client_total_ms") or {}).get("avg", 0.0)),
        float((latency.get("server_total_ms") or {}).get("avg", 0.0)),
        float((latency.get("route_ms") or {}).get("avg", 0.0)),
        float((latency.get("retrieve_ms") or {}).get("avg", 0.0)),
        float((latency.get("fuse_ms") or {}).get("avg", 0.0)),
        float((latency.get("answer_ms") or {}).get("avg", 0.0)),
    ]
    _write_text(
        chart_dir / "latency_avg.svg",
        _bar_chart_svg("Latency Average (ms)", latency_avg_labels, latency_avg_values),
    )

    latency_p95_values = [
        float((latency.get("client_total_ms") or {}).get("p95", 0.0)),
        float((latency.get("server_total_ms") or {}).get("p95", 0.0)),
        float((latency.get("route_ms") or {}).get("p95", 0.0)),
        float((latency.get("retrieve_ms") or {}).get("p95", 0.0)),
        float((latency.get("fuse_ms") or {}).get("p95", 0.0)),
        float((latency.get("answer_ms") or {}).get("p95", 0.0)),
    ]
    _write_text(
        chart_dir / "latency_p95.svg",
        _bar_chart_svg("Latency P95 (ms)", latency_avg_labels, latency_p95_values),
    )

    latency_samples = [float(row.get("client_total_ms") or 0.0) for row in rows]
    _write_text(
        chart_dir / "latency_hist.svg",
        _histogram_svg("Client Latency Distribution (ms)", latency_samples, bins=14),
    )

    metric_labels = [
        "route_hit",
        "fused_hit",
        "evi_recall",
        "answer_ok",
        "exact_match",
        "containment",
    ]
    metric_values = [
        float(retrieval.get("route_hit_rate", 0.0)),
        float(retrieval.get("fused_source_hit_rate", 0.0)),
        float(retrieval.get("avg_evidence_recall_at_k", 0.0)),
        float(answer.get("answer_correct_rate", 0.0)),
        float(answer.get("exact_match_rate", 0.0)),
        float(answer.get("reference_containment_rate", 0.0)),
    ]
    _write_text(
        chart_dir / "retrieval_answer_metrics.svg",
        _bar_chart_svg("Retrieval / Answer Metrics", metric_labels, metric_values, y_as_percent=True),
    )

    sb_labels: List[str] = []
    sb_values: List[float] = []
    for source_type in ("document", "table", "adela"):
        payload = source_breakdown.get(source_type)
        if not payload:
            continue
        sb_labels.append(source_type)
        sb_values.append(float(payload.get("evidence_recall_at_k", 0.0)))
    if not sb_labels:
        sb_labels = ["none"]
        sb_values = [0.0]
    _write_text(
        chart_dir / "source_breakdown.svg",
        _bar_chart_svg("Evidence Recall@K by Expected Source", sb_labels, sb_values, y_as_percent=True),
    )

    html_text = _render_html_report(
        summary=summary,
        rows=rows,
        chart_dir_name="charts",
        top_n=max(args.top_n, 1),
    )
    _write_text(output_dir / "report.html", html_text)

    print("done")
    print(f"summary: {result_dir / 'summary.json'}")
    print(f"rows:    {result_dir / 'requests.jsonl'}")
    print(f"report:  {output_dir / 'report.html'}")
    print(f"charts:  {chart_dir}")


if __name__ == "__main__":
    main()
