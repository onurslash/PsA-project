#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
PsO -> PsA risk simulation + XGBoost + SHAP visualizations (ASCII-safe source)
Usage:
    python3 simulation.py --output-folder ./out [--n 220] [--seed 7] [--dpi 180]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except Exception:
    print("xgboost not found. install via: pip install xgboost", file=sys.stderr)
    raise

try:
    import shap
except Exception:
    print("shap not found. install via: pip install shap", file=sys.stderr)
    raise

TR_LABELS = {
    "Yas": "Ya\u015f",
    "Cinsiyet_Erkek": "Cinsiyet(Erkek=1)",
    "BKI": "BK\u0130",
    "Hastalik_Suresi_yil": "Hastal\u0131k_S\u00fcresi(y\u0131l)",
    "Tirnak_Tutulumu": "T\u0131rnak_Tutulumu",
    "CRP_mgL": "CRP(mg/L)",
    "Sigara": "Sigara",
    "Aile_Oykusu": "Aile_\u00d6yk\u00fc\u00fcs\u00fc(PsO/PsA)",
    "Biyolojik_Tedavi_ge1": "Biyolojik_Tedavi(\u22651)",
    "PASI": "PASI",
    "DLQI": "DLQI",
}

COLS = list(TR_LABELS.keys())
TR_NAMES = [TR_LABELS[c] for c in COLS]


def simulate_data(n=220, seed=7):
    """Create a small simulation dataset (ASCII column names)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Yas": rng.integers(18, 85, size=n),
        "Cinsiyet_Erkek": rng.choice([0, 1], size=n, p=[0.5, 0.5]),
        "BKI": np.clip(rng.normal(27.5, 5.0, size=n), 17, 45),
        "Hastalik_Suresi_yil": rng.integers(0, 25, size=n),
        "Tirnak_Tutulumu": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
        "CRP_mgL": np.clip(rng.normal(5.5, 3.0, size=n), 0.1, 40),
        "Sigara": rng.choice([0, 1], size=n, p=[0.55, 0.45]),
        "Aile_Oykusu": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "Biyolojik_Tedavi_ge1": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "PASI": np.clip(rng.normal(6.0, 4.0, size=n), 0.0, 72.0),
        "DLQI": np.clip(rng.normal(5.0, 4.0, size=n), 0.0, 30.0),
    })[COLS]

    # linear score + noise 
    linpred = (
        0.018*df["Yas"] +
        0.10*df["BKI"] +
        0.55*df["Tirnak_Tutulumu"] +
        0.35*df["CRP_mgL"] +
        0.65*df["Aile_Oykusu"] +
        0.40*df["Hastalik_Suresi_yil"] +
        0.06*df["PASI"] +
        0.04*df["DLQI"] +
        rng.normal(0, 1.0, size=len(df))
    )
    y = (linpred > np.median(linpred)).astype(int)
    return df, y


def train_xgb(X, y):
    model = xgb.XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=0
    )
    model.fit(X, y)
    return model


def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    return explainer, shap_values, expected_value


def save_beeswarm(shap_values, Xvals, out_path, dpi=180):
    plt.figure()
    shap.summary_plot(shap_values, Xvals, feature_names=TR_NAMES, show=False)
    plt.title("K\u00fcresel Etki: SHAP Beeswarm (PsA Tahmini)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_bar_importance(shap_values, Xvals, out_path, dpi=180):
    plt.figure()
    shap.summary_plot(shap_values, Xvals, feature_names=TR_NAMES, plot_type="bar", show=False)
    plt.title("De\u011fi\u015fken \u00d6nem S\u0131ras\u0131 (Ortalama |SHAP|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_dependence_top(shap_values, Xvals, out_path, dpi=180):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    plt.figure()
    shap.dependence_plot(top_idx, shap_values, Xvals, feature_names=TR_NAMES, show=False)
    plt.title("SHAP Ba\u011f\u0131ml\u0131l\u0131k Grafi\u011fi (En Etkili \u00d6zellik)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_waterfall_single(shap_values, X, expected_value, idx, out_path, dpi=180):
    row_shap = shap_values[idx, :]
    x_row = X.iloc[idx, :]
    # build data labels 
    data_labels = []
    for c in COLS:
        val = x_row[c]
        if isinstance(val, (int, np.integer)):
            lab = f"{TR_LABELS[c]} = {val:d}"
        elif isinstance(val, (float, np.floating)):
            lab = f"{TR_LABELS[c]} = {val:.2f}"
        else:
            lab = f"{TR_LABELS[c]} = {val}"
        data_labels.append(lab)

    exp = shap.Explanation(
        values=row_shap,
        base_values=expected_value,
        data=x_row.values,           # numeric values
        feature_names=TR_NAMES       # display names
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=10)
    plt.title("Bireysel Kay\u0131t \u2014 Waterfall (Temsili)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_top10_bar_single(shap_values, X, proba, idx, out_path, dpi=180):
    row_shap = shap_values[idx, :]
    x_row = X.iloc[idx, :]

    k = 10
    order = np.argsort(np.abs(row_shap))[::-1][:k]
    feat_keys = [COLS[i] for i in order]
    vals = row_shap[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ypos = np.arange(len(feat_keys))
    ax.barh(ypos, vals)
    ax.set_yticks(ypos)

    labels = []
    for key in feat_keys:
        val = x_row[key]
        tr = TR_LABELS[key]
        if isinstance(val, (int, np.integer)):
            labels.append(f"{tr} = {val:d}")
        elif isinstance(val, (float, np.floating)):
            labels.append(f"{tr} = {val:.2f}")
        else:
            labels.append(f"{tr} = {val}")
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP katkis\u0131 (log-olasilik)")
    ax.set_title("Bireysel Kay\u0131t \u2014 En B\u00fcy\u00fck 10 Katk\u0131 (|SHAP|)")
    fig.suptitle(f"PsA Tahmin Olas\u0131l\u0131\u011f\u0131: {proba[idx]:.2%}", y=0.99, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PsO->PsA simulation + XGBoost + SHAP visuals (ASCII-safe)")
    parser.add_argument("--output-folder", required=True, help="Output folder")
    parser.add_argument("--n", type=int, default=220, help="Sample size (default: 220)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (default: 7)")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI (default: 180)")
    args = parser.parse_args()

    outdir = args.output_folder
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] simulate: n={args.n}, seed={args.seed}")
    X, y = simulate_data(n=args.n, seed=args.seed)

    csv_path = os.path.join(outdir, "simulasyon_verisi_ascii.csv")
    X.assign(PsA=y).to_csv(csv_path, index=False)
    print(f"[OK] saved data: {csv_path}")

    print("[INFO] train XGBoost ...")
    model = train_xgb(X, y)

    print("[INFO] compute SHAP ...")
    explainer, shap_values, expected_value = compute_shap(model, X)

    proba = model.predict_proba(X)[:, 1]
    idx = int(np.argsort(proba)[-1])

    beeswarm_path = os.path.join(outdir, "01_shap_beeswarm_TR.png")
    bar_path = os.path.join(outdir, "02_shap_bar_TR.png")
    dep_path = os.path.join(outdir, "03_shap_dependence_TOP_TR.png")
    waterfall_path = os.path.join(outdir, "04_shap_waterfall_TR.png")
    singlebar_path = os.path.join(outdir, "05_shap_single_barh_TR.png")

    print("[INFO] beeswarm ...")
    # pass numeric features (values) 
    save_beeswarm(shap_values, X.values, beeswarm_path, dpi=args.dpi)

    print("[INFO] mean |SHAP| bar ...")
    save_bar_importance(shap_values, X.values, bar_path, dpi=args.dpi)

    print("[INFO] dependence (top feature) ...")
    save_dependence_top(shap_values, X.values, dep_path, dpi=args.dpi)

    print("[INFO] waterfall (single record) ...")
    save_waterfall_single(shap_values, X, expected_value, idx, waterfall_path, dpi=args.dpi)

    print("[INFO] single record top-10 bar ...")
    save_top10_bar_single(shap_values, X, proba, idx, singlebar_path, dpi=args.dpi)

    readme_txt = os.path.join(outdir, "README.txt")
    with open(readme_txt, "w", encoding="utf-8") as f:
        f.write(
            "Bu klasorde SHAP tabanli ornek gorseller bulunur.\n"
            "- 01_shap_beeswarm_TR.png : Kuresel etki (beeswarm)\n"
            "- 02_shap_bar_TR.png      : Ortalama |SHAP| onem sirasi\n"
            "- 03_shap_dependence_...  : En etkili ozellik icin bagimlilik grafigi\n"
            "- 04_shap_waterfall_TR.png: Bireysel kayit icin waterfall\n"
            "- 05_shap_single_barh_... : Bireysel kayitta en buyuk 10 katki\n"
            "Not: Veriler temsili simulasyondur; klinik iddia icermez.\n"
        )

    print("\n[OK] generated files:")
    for p in [beeswarm_path, bar_path, dep_path, waterfall_path, singlebar_path, readme_txt, csv_path]:
        print("  -", p)


if __name__ == "__main__":
    main()
