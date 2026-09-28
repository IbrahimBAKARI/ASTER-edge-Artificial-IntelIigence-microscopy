#!/usr/bin/env python3
"""Frozen inter-slide evaluation of the deployed ASTER WBC localizer.

The two slide folders are used only as external test sets. This script never
trains the model and never tunes a threshold on the new slides.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionValidator


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "external_data" / "unseen_slides"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolo" / "wbc_detector.pt"
EXPECTED_WEIGHTS_SHA256 = (
    "5253be6be234b956fa0da9af575949b155b9e47917fb79301d5cb62f8a2d39a1"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SlideData:
    slide_id: str
    source: Path
    images: tuple[Path, ...]
    labels: tuple[Path, ...]
    boxes: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_and_validate(source_root: Path) -> list[SlideData]:
    slides: list[SlideData] = []
    image_hashes: dict[str, Path] = {}
    for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
        images = tuple(
            sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        )
        labels = tuple(
            sorted(path for path in source.glob("*.txt") if path.name != "classes.txt")
        )
        if not images:
            continue
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(
                f"{source.name}: image/label mismatch; "
                f"missing labels={sorted(image_stems - label_stems)}, "
                f"orphan labels={sorted(label_stems - image_stems)}"
            )

        box_count = 0
        for label in labels:
            rows = [row.strip() for row in label.read_text().splitlines() if row.strip()]
            if not rows:
                raise ValueError(f"Empty annotation: {label}")
            for line_number, row in enumerate(rows, 1):
                fields = row.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO row: {label}:{line_number}: {row}")
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
                if class_id != 0:
                    raise ValueError(f"Non-zero class: {label}:{line_number}: {class_id}")
                if not all(0.0 <= value <= 1.0 for value in coords):
                    raise ValueError(f"Out-of-range box: {label}:{line_number}: {row}")
                if coords[2] <= 0.0 or coords[3] <= 0.0:
                    raise ValueError(f"Non-positive box: {label}:{line_number}: {row}")
                box_count += 1

        for image in images:
            digest = sha256(image)
            if digest in image_hashes:
                raise ValueError(f"Duplicate images: {image_hashes[digest]} and {image}")
            image_hashes[digest] = image

        slides.append(
            SlideData(
                slide_id=source.name,
                source=source,
                images=images,
                labels=labels,
                boxes=box_count,
            )
        )
    if len(slides) < 2:
        raise ValueError(f"Expected at least two slide folders under {source_root}")
    return slides


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def stage_dataset(slides: list[SlideData], dataset_root: Path) -> dict[str, Path]:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    yamls: dict[str, Path] = {}
    groups: list[tuple[str, list[SlideData]]] = [
        *((slide.slide_id, [slide]) for slide in slides),
        ("combined_unseen_slides", slides),
    ]
    for group_name, group_slides in groups:
        group_root = dataset_root / group_name
        image_dir = group_root / "images" / "test"
        label_dir = group_root / "labels" / "test"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for slide in group_slides:
            for image in slide.images:
                prefix = "" if len(group_slides) == 1 else f"{slide.slide_id}__"
                replace_symlink(image_dir / f"{prefix}{image.name}", image)
                replace_symlink(label_dir / f"{prefix}{image.stem}.txt", slide.source / f"{image.stem}.txt")
        data_yaml = group_root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(group_root),
                    "train": "images/test",
                    "val": "images/test",
                    "test": "images/test",
                    "names": {0: "wbc"},
                },
                sort_keys=False,
            )
        )
        yamls[group_name] = data_yaml
    return yamls


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


class CapturingDetectionValidator(DetectionValidator):
    """Keep the standard validator instance so exact TP/FP/FN remain auditable."""

    latest: "CapturingDetectionValidator | None" = None
    captured_stats: dict[str, np.ndarray]

    def get_stats(self) -> dict[str, object]:
        self.captured_stats = {
            key: np.concatenate(values, axis=0)
            for key, values in self.metrics.stats.items()
            if values
        }
        return super().get_stats()

    def __call__(self, *args, **kwargs):
        metrics = super().__call__(*args, **kwargs)
        type(self).latest = self
        return metrics


def run_standard_validator(
    model: YOLO,
    data_yaml: Path,
    tag: str,
    device: str,
    batch: int,
    confidence: float,
    runs_root: Path,
) -> tuple[dict[str, float], dict[str, int]]:
    result = model.val(
        validator=CapturingDetectionValidator,
        data=str(data_yaml),
        split="test",
        imgsz=960,
        batch=batch,
        device=device,
        workers=0,
        conf=confidence,
        iou=0.50,
        classes=[0],
        max_det=300,
        rect=True,
        half=False,
        augment=False,
        plots=False,
        verbose=False,
        save_json=False,
        project=str(runs_root),
        name=tag,
        exist_ok=True,
    )
    validator = CapturingDetectionValidator.latest
    if validator is None:
        raise RuntimeError("The standard validator was not captured")
    stats = validator.captured_stats
    tp = int(stats["tp"][:, 0].sum())
    predictions = int(len(stats["conf"]))
    references = int(len(stats["target_cls"]))
    counts = {
        "reference_wbc": references,
        "detections": predictions,
        "tp": tp,
        "fp": predictions - tp,
        "fn": references - tp,
    }
    metrics = {
        "precision_at_evaluator_best_f1": float(result.box.mp),
        "recall_at_evaluator_best_f1": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }
    return metrics, counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_figure(metric_rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    per_slide = [row for row in metric_rows if row["dataset"] != "combined_unseen_slides"]
    historical = {
        "map50": 0.9823130169433179,
        "precision_conf018": 164 / 183,
        "recall_conf018": 164 / 171,
    }
    plot_rows = [historical, *per_slide]
    labels = [
        "Original within-slide\nevaluation",
        *[f"Unseen slide\n{str(row['dataset']).replace('captures_Lame', '')}" for row in per_slide],
    ]
    series = [
        ("mAP@50", "map50", "#2F6B9A"),
        ("Precision (conf=0.18)", "precision_conf018", "#3C9D77"),
        ("Recall (conf=0.18)", "recall_conf018", "#D67B32"),
    ]
    x = np.arange(len(labels))
    width = 0.23
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, (label, key, color) in enumerate(series):
        values = [float(row[key]) for row in plot_rows]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Frozen localizer: within-slide baseline and unseen-slide transfer")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(metric_rows: list[dict[str, object]], output: Path) -> None:
    combined = next(row for row in metric_rows if row["dataset"] == "combined_unseen_slides")
    table = [
        "| Test set | Fields | Reference WBC | mAP@50 | mAP@50–95 | TP | FP | FN | Precision¹ | Recall¹ | F1¹ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| Original slide, within-slide evaluation² | 104 | 171 | 0.9823 | 0.6598 | 164 | 19 | 7 | 0.8962 | 0.9591 | 0.9266 |",
    ]
    for row in metric_rows:
        table.append(
            f"| {row['dataset']} | {row['images']} | {row['reference_wbc']} | "
            f"{float(row['map50']):.4f} | {float(row['map50_95']):.4f} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | "
            f"{float(row['precision_conf018']):.4f} | {float(row['recall_conf018']):.4f} | "
            f"{float(row['f1_conf018']):.4f} |"
        )
    output.write_text(
        "# Frozen inter-slide generalization test\n\n"
        "The historical 960-pixel adapted localizer was frozen and evaluated on two "
        "additional slides that were never used for training, validation, threshold selection, "
        "or model selection. The model checksum is recorded in `provenance.json`.\n\n"
        + "\n".join(table)
        + "\n\n¹ Fixed deployment operating point: confidence 0.18, NMS IoU 0.50; "
        "TP matching IoU ≥ 0.50. The AP metrics are confidence-integrated Ultralytics metrics.\n\n"
        "² Historical within-slide test result, included only as a comparator; it is not an "
        "external-slide evaluation. Sources: `resolution_ablation_v1/0708f017730c31a8/` and "
        "`ASTER_research_archive/.../phase6_campaign_x40/yield/threshold_sweep.csv`.\n\n"
        "## Result suitable for the manuscript\n\n"
        f"> The frozen localizer was evaluated without retraining or threshold adjustment on "
        f"two independently acquired, entirely unseen educational blood-smear slides comprising "
        f"{combined['images']} fields and {combined['reference_wbc']} annotated WBCs. On the pooled "
        f"external slides, it achieved mAP@50={float(combined['map50']):.3f} and "
        f"mAP@50–95={float(combined['map50_95']):.3f}. At the prespecified deployment operating "
        f"point (confidence 0.18; NMS IoU 0.50), precision was "
        f"{float(combined['precision_conf018']):.3f} and recall was "
        f"{float(combined['recall_conf018']):.3f} ({combined['tp']} TP, "
        f"{combined['fp']} FP, and {combined['fn']} FN).\n\n"
        "## Interpretation boundary\n\n"
        "This experiment tests transfer to two unseen slides acquired with the same ASTER setup. "
        "It is evidence of preliminary inter-slide generalization. Two slides remain a limited "
        "sample and do not establish robustness across laboratories, staining protocols, devices, "
        "or operators.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    arguments = parser.parse_args()

    if not arguments.weights.is_file():
        raise FileNotFoundError(arguments.weights)
    weights_hash = sha256(arguments.weights)
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(
            f"Wrong checkpoint: expected {EXPECTED_WEIGHTS_SHA256}, found {weights_hash}"
        )

    slides = discover_and_validate(arguments.source)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_yamls = stage_dataset(slides, ROOT / "dataset_staging")
    model = YOLO(str(arguments.weights), task="detect")

    metric_rows: list[dict[str, object]] = []
    slide_by_name = {slide.slide_id: slide for slide in slides}
    for dataset_name, data_yaml in data_yamls.items():
        print(f"Standard AP evaluation: {dataset_name}", flush=True)
        metrics, _ = run_standard_validator(
            model,
            data_yaml,
            f"{dataset_name}_ap",
            arguments.device,
            arguments.batch,
            0.001,
            output / "ultralytics_runs",
        )
        print(f"Fixed operating point: {dataset_name}", flush=True)
        _, counts = run_standard_validator(
            model,
            data_yaml,
            f"{dataset_name}_conf018",
            arguments.device,
            arguments.batch,
            0.18,
            output / "ultralytics_runs",
        )
        precision = safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])
        image_count = (
            sum(len(slide.images) for slide in slides)
            if dataset_name == "combined_unseen_slides"
            else len(slide_by_name[dataset_name].images)
        )
        metric_rows.append(
            {
                "dataset": dataset_name,
                "images": image_count,
                **counts,
                "precision_conf018": precision,
                "recall_conf018": recall,
                "f1_conf018": f1,
                **metrics,
            }
        )
    stale_per_image = output / "per_image_operating_point.csv"
    if stale_per_image.exists():
        stale_per_image.unlink()
    write_csv(output / "inter_slide_metrics.csv", metric_rows)
    create_figure(metric_rows, output / "inter_slide_metrics.png")
    write_report(metric_rows, output / "REPORT_INTER_SLIDE.md")

    provenance = {
        "schema": "aster-inter-slide-generalization-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "training": False,
            "threshold_tuning_on_external_slides": False,
            "weights": str(arguments.weights.resolve().relative_to(REPO_ROOT)) if arguments.weights.resolve().is_relative_to(REPO_ROOT) else arguments.weights.name,
            "weights_sha256": weights_hash,
            "adaptation_imgsz": 960,
            "evaluation_imgsz": 960,
            "ap_confidence_floor": 0.001,
            "deployment_confidence": 0.18,
            "nms_iou": 0.50,
            "tp_iou": 0.50,
            "class_ids": [0],
            "device": arguments.device,
            "batch": arguments.batch,
            "validator": "Ultralytics DetectionValidator",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "numpy": np.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "slides": [
            {
                "slide_id": slide.slide_id,
                "source": str(Path("external_data/unseen_slides") / slide.slide_id),
                "images": len(slide.images),
                "reference_wbc": slide.boxes,
                "image_names_sha256": hashlib.sha256(
                    "\n".join(image.name for image in slide.images).encode()
                ).hexdigest(),
                "label_contents_sha256": hashlib.sha256(
                    b"".join(label.read_bytes() for label in slide.labels)
                ).hexdigest(),
            }
            for slide in slides
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
