import os
import glob
import configparser
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import joblib
import shap


def _read_project_path(config_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    for section in ("General settings", "Project"):
        if cfg.has_section(section) and cfg.has_option(section, "project_path"):
            p = cfg.get(section, "project_path", fallback="").strip()
            if p:
                return os.path.abspath(p)
    return os.path.abspath(os.path.dirname(config_path))


def _get_labels(cfg: configparser.ConfigParser) -> List[str]:
    labels = []
    if cfg.has_option("SML settings", "no_targets"):
        try:
            n = int(cfg.get("SML settings", "no_targets"))
        except Exception:
            n = 0
        for i in range(1, n + 1):
            key = f"target_name_{i}"
            if cfg.has_option("SML settings", key):
                v = cfg.get("SML settings", key).strip()
                if v:
                    labels.append(v)
    return labels


def _get_model_dir(cfg: configparser.ConfigParser) -> Optional[str]:
    if cfg.has_section("SML settings") and cfg.has_option("SML settings", "model_dir"):
        v = cfg.get("SML settings", "model_dir", fallback="").strip()
        return os.path.abspath(v) if v else None
    return None


def _load_targets_inserted(project_path: str, video_stem: Optional[str]) -> pd.DataFrame:
    targets_dir = os.path.join(project_path, "csv", "targets_inserted")
    if not os.path.isdir(targets_dir):
        raise FileNotFoundError(f"targets_inserted folder not found: {targets_dir}")

    pattern = os.path.join(targets_dir, f"{video_stem}.csv") if video_stem else os.path.join(targets_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No targets_inserted CSVs found for pattern: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _pick_samples(
    df: pd.DataFrame,
    label: str,
    n_present: int,
    n_absent: int,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.Series]:
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in targets_inserted CSVs")

    y = df[label].astype(int)
    present_idx = np.where(y.values == 1)[0]
    absent_idx = np.where(y.values == 0)[0]

    if len(present_idx) == 0 or len(absent_idx) == 0:
        raise ValueError(f"Label '{label}' has no present or absent frames in targets_inserted data.")

    n_present = min(n_present, len(present_idx))
    n_absent = min(n_absent, len(absent_idx))

    present_sel = rng.choice(present_idx, size=n_present, replace=False)
    absent_sel = rng.choice(absent_idx, size=n_absent, replace=False)

    sel = np.concatenate([present_sel, absent_sel])
    df_sel = df.iloc[sel].reset_index(drop=True)
    y_sel = y.iloc[sel].reset_index(drop=True)
    return df_sel, y_sel


def _build_explainer(model, background: pd.DataFrame):
    try:
        return shap.TreeExplainer(model, data=background, model_output="probability")
    except Exception:
        return shap.TreeExplainer(model, data=background)


def compute_simba_shap(
    config_path: str,
    *,
    labels: Optional[List[str]] = None,
    video_stem: Optional[str] = None,
    n_present: int = 200,
    n_absent: int = 200,
    out_dir: Optional[str] = None,
    random_state: int = 7,
) -> List[str]:
    """
    Compute SHAP values for SimBA RF models using targets_inserted samples.

    Returns list of generated output file paths.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SimBA config not found: {config_path}")

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    project_path = _read_project_path(config_path)
    model_dir = _get_model_dir(cfg)
    if not model_dir:
        raise ValueError("model_dir not set in SimBA config [SML settings]")

    labels = labels or _get_labels(cfg)
    if not labels:
        raise ValueError("No labels found in SimBA config (SML settings target_name_*)")

    df = _load_targets_inserted(project_path, video_stem=video_stem)

    # Feature columns = everything except labels and common metadata
    drop_cols = set(labels) | {"frame", "video", "Video"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    out_dir = out_dir or os.path.join(project_path, "shap")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(random_state)
    outputs = []

    def _get_model_feature_names(model) -> Optional[List[str]]:
        if hasattr(model, "feature_names_in_"):
            try:
                return list(model.feature_names_in_)
            except Exception:
                pass
        return None

    for label in labels:
        model_path = os.path.join(model_dir, f"{label}.sav")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        model_feats = _get_model_feature_names(model)

        df_sel, y_sel = _pick_samples(df, label, n_present, n_absent, rng)
        if model_feats:
            missing = [c for c in model_feats if c not in df_sel.columns]
            if missing:
                raise ValueError(
                    f"Model '{label}' expects {len(model_feats)} features, "
                    f"but {len(missing)} are missing in targets_inserted. "
                    f"Example missing: {missing[:10]}"
                )
            X = df_sel[model_feats].copy()
        else:
            X = df_sel[feature_cols].copy()

        # Background: small random subset
        bg = X.sample(min(200, len(X)), random_state=random_state)

        explainer = _build_explainer(model, bg)

        # SHAP values
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            # binary classifiers => use positive class
            shap_vals = shap_values[1]
            exp_val = expected_value[1]
        else:
            shap_vals = shap_values
            exp_val = expected_value

        proba = model.predict_proba(X)[:, 1]
        model_out = shap_vals.sum(axis=1) + exp_val

        shap_df = pd.DataFrame(shap_vals, columns=X.columns)
        shap_df.insert(0, "expected_value", exp_val)
        shap_df.insert(1, "model_output", model_out)
        shap_df.insert(2, "prediction_probability", proba)
        shap_df.insert(3, "target_present", y_sel.values)

        raw_df = X.copy()
        raw_df.insert(0, "target_present", y_sel.values)

        shap_csv = os.path.join(out_dir, f"{label}_shap_values.csv")
        raw_csv = os.path.join(out_dir, f"{label}_shap_feature_values.csv")
        shap_df.to_csv(shap_csv, index=False)
        raw_df.to_csv(raw_csv, index=False)
        outputs.extend([shap_csv, raw_csv])

        # Summary plots
        plt.figure()
        shap.summary_plot(shap_vals, X, show=False)
        summary_png = os.path.join(out_dir, f"{label}_shap_summary.png")
        plt.tight_layout()
        plt.savefig(summary_png, dpi=200)
        plt.close()
        outputs.append(summary_png)

        plt.figure()
        shap.summary_plot(shap_vals, X, show=False, plot_type="bar")
        bar_png = os.path.join(out_dir, f"{label}_shap_bar.png")
        plt.tight_layout()
        plt.savefig(bar_png, dpi=200)
        plt.close()
        outputs.append(bar_png)

    return outputs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute SHAP for SimBA models")
    p.add_argument("--config", required=True, help="Path to SimBA project_config.ini")
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Labels (space- or comma-separated). Example: --labels groom,rest or --labels groom rest",
    )
    p.add_argument("--video-stem", default="", help="Optional video stem to filter targets_inserted")
    p.add_argument("--present", type=int, default=200)
    p.add_argument("--absent", type=int, default=200)
    p.add_argument("--out", default="", help="Output dir (optional)")
    args = p.parse_args()

    lab = None
    if args.labels:
        if len(args.labels) == 1:
            lab = [x.strip() for x in args.labels[0].split(",") if x.strip()]
        else:
            lab = []
            for item in args.labels:
                lab.extend([x.strip() for x in item.split(",") if x.strip()])
    outputs = compute_simba_shap(
        args.config,
        labels=lab,
        video_stem=args.video_stem or None,
        n_present=args.present,
        n_absent=args.absent,
        out_dir=args.out or None,
    )
    print("\n".join(outputs))
