import argparse
import os
import configparser
from typing import List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimBA RF models for one or more labels")
    parser.add_argument("--config", required=True, help="Path to SimBA project_config.ini")
    parser.add_argument("--labels", default="", help="Comma-separated labels to train (optional)")
    return parser.parse_args()


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


def _normalize_model_dir(model_dir: str) -> str:
    if not model_dir:
        return model_dir
    norm = os.path.normpath(model_dir)
    parts = norm.split(os.sep)
    if len(parts) >= 2 and parts[-1] == "generated_models" and parts[-2] == "generated_models":
        norm = os.path.dirname(norm)
    return norm


def _ensure_model_paths(cfg: configparser.ConfigParser, labels: List[str]) -> None:
    model_dir = cfg.get("SML settings", "model_dir", fallback="").strip()
    model_dir = _normalize_model_dir(model_dir)
    if not model_dir:
        return
    cfg.set("SML settings", "model_dir", model_dir)
    os.makedirs(model_dir, exist_ok=True)
    for i, label in enumerate(labels, start=1):
        key = f"model_path_{i}"
        expected = os.path.join(model_dir, f"{label}.sav")
        if not cfg.has_option("SML settings", key) or cfg.get("SML settings", key).strip() != expected:
            cfg.set("SML settings", key, expected)


def _clean_targets_inserted(cfg: configparser.ConfigParser) -> None:
    project_path = cfg.get("Project", "project_path", fallback="").strip()
    if not project_path:
        return
    targets_dir = os.path.join(project_path, "csv", "targets_inserted")
    if not os.path.isdir(targets_dir):
        return
    for fn in os.listdir(targets_dir):
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(targets_dir, fn)
        try:
            df = pd.read_csv(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
                df.to_csv(path, index=False)
        except Exception:
            continue


def main() -> int:
    args = _parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    labels = []
    if args.labels.strip():
        labels = [t.strip() for t in args.labels.split(",") if t.strip()]
    if not labels:
        labels = _get_labels(cfg)
    if not labels:
        raise RuntimeError("No labels found. Set SML settings target_name_* in the config.")

    _ensure_model_paths(cfg, labels)
    _clean_targets_inserted(cfg)

    # Save config updates (model paths)
    with open(config_path, "w", encoding="utf-8") as f:
        cfg.write(f)

    from simba.model.train_rf import TrainRandomForestClassifier

    original_classifier = cfg.get("create ensemble settings", "classifier", fallback="")
    for label in labels:
        cfg.set("create ensemble settings", "classifier", label)
        with open(config_path, "w", encoding="utf-8") as f:
            cfg.write(f)
        print(f"[SimBA] Training label: {label}")
        trainer = TrainRandomForestClassifier(config_path=config_path)
        trainer.run()
        trainer.save()

    if original_classifier:
        cfg.set("create ensemble settings", "classifier", original_classifier)
        with open(config_path, "w", encoding="utf-8") as f:
            cfg.write(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
