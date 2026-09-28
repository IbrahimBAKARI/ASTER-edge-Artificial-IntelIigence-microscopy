#!/usr/bin/env python3
"""Verify the compact V1.1.0 experiment artefacts used in the manuscript."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    resolution = read_csv(ROOT / "resolution_ablation/results/metrics_all.csv")

    def map50(model: str, dataset: str, evaluation_size: int) -> float:
        rows = [
            row
            for row in resolution
            if row["model"] == model
            and row["dataset"] == dataset
            and row["eval_imgsz"] == str(evaluation_size)
        ]
        if len(rows) != 1:
            raise AssertionError((model, dataset, evaluation_size, len(rows)))
        return float(rows[0]["map50"])

    assert abs(map50("initial_640", "x40_test", 640) - 0.37224348996009793) < 1e-12
    assert abs(map50("adapted_640", "x40_test", 640) - 0.9679143224307011) < 1e-12
    assert abs(map50("adapted_960", "x40_test", 960) - 0.9823130169433179) < 1e-12

    ood = read_csv(ROOT / "ood_progressive_degradation/results/ood_summary.csv")
    assert len(ood) == 4
    assert {row["family"] for row in ood} == {"blur", "color"}

    inter_slide = read_csv(ROOT / "inter_slide_generalization/results/inter_slide_metrics.csv")
    pooled = next(row for row in inter_slide if row["dataset"] == "combined_unseen_slides")
    assert int(pooled["images"]) == 159
    assert int(pooled["reference_wbc"]) == 241
    assert abs(float(pooled["map50"]) - 0.90384294720932) < 1e-12
    assert (int(pooled["tp"]), int(pooled["fp"]), int(pooled["fn"])) == (185, 13, 56)

    print("V1.1.0 experiment artefacts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
