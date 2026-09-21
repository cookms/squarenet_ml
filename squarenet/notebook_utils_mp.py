"""Small notebook-facing utilities that are not central to the scientific story."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def prepare_targets(frame: pd.DataFrame, config: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    rows: list[dict[str, Any]] = []
    regression_specs = {
        "target_band_gap_ev": ("Band gap (eV)", "mp__band_gap_ev"),
        "target_energy_above_hull_ev_atom": ("Energy above hull (eV/atom)", "mp__energy_above_hull_ev_atom"),
        "target_formation_energy_per_atom_ev": ("Formation energy per atom (eV/atom)", "mp__formation_energy_per_atom_ev"),
    }
    for label, (name, source) in regression_specs.items():
        data[label] = pd.to_numeric(data[source], errors="coerce") if source in data else np.nan
        rows.append({"target": label, "display_name": name, "task": "regression"})

    rows.extend([
        {"target": "target_is_metal", "display_name": "Metal (1) vs nonmetal (0)", "task": "classification"},
        {
            "target": "target_near_hull",
            "display_name": f"Near-hull stability (<= {config.near_hull_threshold_ev_atom:.2f} eV/atom)",
            "task": "classification",
        },
    ])
    data["target_is_metal"] = data["mp__is_metal"].astype("Int64")
    if "mp__is_stable" in data:
        data["target_is_stable"] = data["mp__is_stable"].astype("Int64")
    hull = pd.to_numeric(data["mp__energy_above_hull_ev_atom"], errors="coerce")
    data["target_near_hull"] = (hull <= config.near_hull_threshold_ev_atom).where(hull.notna()).astype("Int64")

    registry = pd.DataFrame(rows)
    registry["n_non_missing"] = registry["target"].map(lambda c: int(data[c].notna().sum()))
    registry["n_missing"] = registry["target"].map(lambda c: int(data[c].isna().sum()))
    registry["n_unique"] = registry["target"].map(lambda c: int(data[c].nunique(dropna=True)))
    return data, registry


def target_audit_table(data: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    audited = registry.copy()
    audited["n_non_missing"] = audited["target"].map(lambda c: int(data[c].notna().sum()))
    audited["n_groups"] = audited["target"].map(lambda c: int(data.loc[data[c].notna(), "cv_group"].nunique()))
    audited["positive_rate"] = audited.apply(
        lambda row: float(pd.to_numeric(data[row["target"]], errors="coerce").mean())
        if row["task"] == "classification" and row["n_non_missing"] else np.nan,
        axis=1,
    )
    return audited


def save_figure(fig: plt.Figure, name: str, figure_dir: Path | None = None) -> Path:
    if figure_dir is None:
        from .experiment_config import FIGURE_DIR
        figure_dir = FIGURE_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def plot_binned_target(df: pd.DataFrame, feature: str, target: str, *, bins: int = 10):
    data = df[[feature, target]].dropna().copy()
    data["feature_bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    summary = data.groupby("feature_bin", observed=True).agg(
        feature_mean=(feature, "mean"), target_mean=(target, "mean"),
        target_std=(target, "std"), count=(target, "size"),
    ).reset_index(drop=True)
    summary["standard_error"] = summary["target_std"] / np.sqrt(summary["count"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(summary["feature_mean"], summary["target_mean"], yerr=summary["standard_error"], fmt="o-")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"Binned relationship: {feature} vs {target}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def columns_matching(columns: Sequence[str], patterns: Sequence[str]) -> list[str]:
    return [c for c in columns if any(re.search(pattern, c, flags=re.IGNORECASE) for pattern in patterns)]


def material_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    aggregated = predictions.groupby(
        ["target","target_name","task","feature_set","material_id","cv_group","formula"], as_index=False
    ).agg(y_true=("y_true","first"), mean_y_pred=("y_pred","mean"), mean_y_score=("y_score","mean"), n_oof=("repeat","nunique"))
    regression = aggregated[aggregated.task == "regression"].copy()
    regression["error"] = (regression.y_true - regression.mean_y_pred).abs()
    classification = aggregated[aggregated.task == "classification"].copy()
    classification["error"] = (classification.y_true - classification.mean_y_score) ** 2
    regression["mean_prediction"] = regression["mean_y_pred"]
    classification["mean_prediction"] = classification["mean_y_score"]
    combined = pd.concat([regression, classification], ignore_index=True)
    baseline = combined[combined.feature_set == "Baseline"][["target","material_id","error"]].rename(columns={"error":"baseline_error"})
    candidates = combined[combined.feature_set != "Baseline"].copy().merge(baseline, on=["target","material_id"], how="left", validate="many_to_one")
    candidates["material_error_improvement"] = candidates["baseline_error"] - candidates["error"]
    return candidates
