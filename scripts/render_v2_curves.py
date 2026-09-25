"""Render the checked-in v2 development-loss curves as an accessible SVG."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#1769aa", "#cb5b32", "#18836b")


def render(summary: dict) -> str:
    runs = summary["runs"]
    if len(runs) != 3 or any(len(run["dev_loss_by_epoch"]) != 3 for run in runs):
        raise ValueError("expected three three-epoch development curves")
    all_losses = [point["loss"] for run in runs for point in run["dev_loss_by_epoch"]]
    low = max(0, min(all_losses) - 0.35)
    high = max(all_losses) + 0.35
    left, right, top, bottom = 96, 824, 82, 390

    def x(epoch: int) -> float:
        return left + (epoch - 1) * (right - left) / 2

    def y(loss: float) -> float:
        return bottom - (loss - low) * (bottom - top) / (high - low)

    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 510" role="img" '
            'aria-labelledby="chart-title chart-desc">'
        ),
        '<title id="chart-title">ExitReceipt v2 development loss across three epochs</title>',
        (
            '<desc id="chart-desc">Three fixed LoRA seeds. The selected adapter is chosen by '
            "lowest development loss, not test accuracy.</desc>"
        ),
        '<rect width="920" height="510" fill="#f7f9f7"/>',
        (
            '<text x="96" y="38" fill="#142b24" font-family="system-ui" font-size="23" '
            'font-weight="700">How long should the adapter train?</text>'
        ),
        (
            '<text x="96" y="61" fill="#5b6d63" font-family="system-ui" font-size="13">'
            "Development loss · lower is better · test excluded from selection</text>"
        ),
    ]
    for tick in range(5):
        value = low + tick * (high - low) / 4
        yy = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" '
            'stroke="#dce4df" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 16}" y="{yy + 4:.1f}" text-anchor="end" fill="#52645a" '
            f'font-family="ui-monospace,monospace" font-size="12">{value:.1f}</text>'
        )
    for epoch in range(1, 4):
        xx = x(epoch)
        parts.append(
            f'<text x="{xx:.1f}" y="{bottom + 25}" text-anchor="middle" fill="#52645a" '
            f'font-family="ui-monospace,monospace" font-size="12">{epoch}</text>'
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 50}" text-anchor="middle" '
        'fill="#52645a" font-family="system-ui" font-size="13">Epoch</text>'
    )
    for index, run in enumerate(runs):
        color = COLORS[index]
        points = run["dev_loss_by_epoch"]
        coordinates = " ".join(f"{x(p['epoch']):.1f},{y(p['loss']):.1f}" for p in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        best = min(points, key=lambda point: point["loss"])
        for point in points:
            radius = 7 if point is best else 4
            parts.append(
                f'<circle cx="{x(point["epoch"]):.1f}" cy="{y(point["loss"]):.1f}" '
                f'r="{radius}" fill="{color}" stroke="#f7f9f7" stroke-width="2"/>'
            )
        parts.append(f'<circle cx="{124 + index * 240}" cy="473" r="6" fill="{color}"/>')
        marker = " · published" if run["seed"] == summary["selected_seed"] else ""
        parts.append(
            f'<text x="{139 + index * 240}" y="478" fill="#263e32" font-family="system-ui" '
            f'font-size="13">Seed {run["seed"]}{marker}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    summary = json.loads((ROOT / "results/v2-robustness.json").read_text(encoding="utf-8"))
    output = ROOT / "docs/assets/v2-dev-loss.svg"
    output.write_text(render(summary), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
