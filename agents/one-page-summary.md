# One-page summary workflow (QPS vs Recall)

This doc explains how the one-page summary in `benchmark_results/qps_recall_summary.md` was produced so it can be replicated for future runs.

## Inputs

- Required: one or more benchmark run folders, each containing `all_results.json` and `*_config.yaml` files.
- This summary used:
  - `benchmark_results/benchmark_20251204_173318/benchmark_summary.md`
  - `benchmark_results/benchmark_20251204_152957/benchmark_summary.md`
  - `benchmark_results/benchmark_20251204_173318/all_results.json`
  - `benchmark_results/benchmark_20251204_152957/all_results.json`
  - `benchmark_results/benchmark_20251204_173318/random/random_config.yaml`
  - `benchmark_results/benchmark_20251204_173318/glove50/glove50_config.yaml`

## Process overview

1) **Read metrics**
   - Use each run’s `all_results.json` to extract per‑algorithm `qps` and `recall` for each dataset.
   - Keep algorithm names as the plot labels.

2) **Plot QPS vs recall**
   - Make one plot per dataset (e.g., `random`, `glove50`) from a **single run folder**.
   - X‑axis: QPS on a log scale.
   - Y‑axis: recall for `topk` (these runs used `topk=20`).
   - Use **numbered point annotations** to avoid label overlap:
     - Each point gets a small index near the marker.
     - A right‑side vertical list maps index → algorithm name.
     - No leader lines or timestamp legend.
   - Output SVGs to `benchmark_results/`:
     - `qps_recall_random.svg`
     - `qps_recall_glove50.svg`

3) **Collect algorithm implementation details**
   - Read the dataset config YAMLs (`*_config.yaml`) to list algorithm definitions:
     - Indexer/searcher class names
     - Key parameters (e.g., `HNSWIndexer` M/ef, Faiss index keys, `nprobe`, LSH bits)
   - Include common run settings (metric, topk, n_queries, repeat).

4) **Collect dataset details**
   - Read dataset settings from config YAML:
     - Dataset name, dimensions, train size/limit, test size, queries, ground_truth_k, dataset seed.

5) **Write one-page summary**
   - Create `benchmark_results/qps_recall_summary.md` with sections:
     - Plots (embedded SVGs)
     - Algorithms and implementation details
     - Dataset details
     - Brief takeaways based on the plots

## Example plotting script (SVG only)

The environment lacked plotting libraries, so a minimal Python script generated SVG directly. This is sufficient for QPS‑recall scatter plots and avoids extra dependencies.

Key points to implement:
- Parse a **single** `all_results.json`.
- Use log‑scale on QPS.
- Draw axes, ticks, labels, and points.
- Add numeric indices next to points, and a right‑side list mapping indices → algorithm names.
- Save to `benchmark_results/qps_recall_<dataset>.svg`.

### Minimal SVG generator (template)

Use this as a template and fill in the data parsing section:

```python
import json, math
from pathlib import Path

def draw_svg(dataset, points, out_path):
    width, height = 880, 440
    margin = dict(left=70, right=220, top=40, bottom=55)
    plot_w = width - margin['left'] - margin['right']
    plot_h = height - margin['top'] - margin['bottom']

    x_vals = [p['qps'] for p in points]
    y_vals = [p['recall'] for p in points]
    x_min, x_max = min(x_vals), max(x_vals)
    x_min = max(x_min * 0.8, 1e-3)
    x_max = x_max * 1.2
    y_min, y_max = 0.35, 1.02

    def x_to_px(x):
        lx = math.log10(x)
        lmin = math.log10(x_min)
        lmax = math.log10(x_max)
        return margin['left'] + (lx - lmin) / (lmax - lmin) * plot_w

    def y_to_px(y):
        return margin['top'] + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    points = sorted(points, key=lambda p: (-p['recall'], -p['qps'], p['algo']))
    for i, p in enumerate(points, start=1):
        p['idx'] = i
        p['x'] = x_to_px(p['qps'])
        p['y'] = y_to_px(p['recall'])

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<style>text{font-family:Arial,sans-serif;fill:#111}.axis{stroke:#111}.grid{stroke:#999;stroke-dasharray:3 3;opacity:.4}.label{font-size:11px}.title{font-size:14px;font-weight:600}.bg{fill:#fff}.index{font-size:10px;font-weight:600}</style>')
    lines.append('<rect class="bg" x="0" y="0" width="100%" height="100%"/>')

    x0, y0 = margin['left'], margin['top'] + plot_h
    x1, y1 = margin['left'] + plot_w, margin['top']
    lines.append(f'<text class="title" x="{margin["left"]}" y="22">QPS vs Recall — {dataset}</text>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')

    tick_vals = [1, 10, 100, 1000, 10000, 100000]
    for tv in tick_vals:
        if tv < x_min or tv > x_max:
            continue
        tx = x_to_px(tv)
        lines.append(f'<line class="grid" x1="{tx}" y1="{y0}" x2="{tx}" y2="{y1}"/>')
        label = f'{int(tv):,}' if tv >= 1 else f'{tv:g}'
        lines.append(f'<text class="label" x="{tx}" y="{y0 + 18}" text-anchor="middle">{label}</text>')

    y_ticks = [0.4, 0.6, 0.8, 1.0]
    for tv in y_ticks:
        ty = y_to_px(tv)
        lines.append(f'<line class="grid" x1="{x0}" y1="{ty}" x2="{x1}" y2="{ty}"/>')
        lines.append(f'<text class="label" x="{x0 - 10}" y="{ty + 4}" text-anchor="end">{tv:.1f}</text>')

    lines.append(f'<text class="label" x="{(x0 + x1) / 2}" y="{height - 15}" text-anchor="middle">QPS (log scale)</text>')
    lines.append(f'<text class="label" x="16" y="{(y0 + y1) / 2}" transform="rotate(-90 16 {(y0 + y1) / 2})" text-anchor="middle">Recall (topk=20)</text>')

    for p in points:
        lines.append(f'<circle cx="{p["x"]:.2f}" cy="{p["y"]:.2f}" r="4.2" fill="#d62728" stroke="#111" stroke-width="0.6"/>')
        lines.append(f'<text class="index" x="{p["x"] + 6:.2f}" y="{p["y"] - 4:.2f}">{p["idx"]}</text>')

    list_x = x1 + 16
    list_y = y1 + 6
    line_h = 14
    lines.append(f'<text class="label" x="{list_x}" y="{list_y}">Labels</text>')
    list_y += 14
    for p in points:
        lines.append(f'<text class="label" x="{list_x}" y="{list_y}">{p["idx"]}. {p["algo"]}</text>')
        list_y += line_h

    lines.append('</svg>')
    out_path.write_text('\\n'.join(lines))

# Example: build points from a single run's all_results.json
data = json.loads(Path('benchmark_results/benchmark_20251204_173318/all_results.json').read_text())
for dataset in ['random', 'glove50']:
    points = []
    for algo, metrics in data.get(dataset, {}).items():
        if metrics.get('qps') is None or metrics.get('recall') is None:
            continue
        points.append({'algo': algo, 'qps': float(metrics['qps']), 'recall': float(metrics['recall'])})
    draw_svg(dataset, points, Path(f'benchmark_results/qps_recall_{dataset}.svg'))
```

## Naming and locations

- Output summary: `benchmark_results/qps_recall_summary.md`
- Plot assets: `benchmark_results/qps_recall_<dataset>.svg`

## Checklist for future runs

- [ ] Identify run folders and list datasets
- [ ] Extract `qps` and `recall` per algorithm from a single `all_results.json`
- [ ] Generate QPS‑recall plots (per dataset)
- [ ] Pull algorithm config details from `*_config.yaml`
- [ ] Pull dataset sizes and dimensions from `*_config.yaml`
- [ ] Write the one‑page summary markdown with plots + details + takeaways
