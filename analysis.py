"""
Clean the kill-feature table, compare own vs pro by weapon class,
plot distributions, and fit an unsupervised anomaly detector per class.

    pip install pandas numpy matplotlib scikit-learn
    python analysis.py [kill_features.csv]

Outputs: plots_*.png, anomalies.csv
Anomaly scores here mean "unusual relative to this dataset", NOT "cheating".
`flagged` marks the top 2% by construction (contamination=0.02), so the overall
flagged share is 2% whatever the data looks like; only the split between groups is
informative.

PRIVACY: anomalies.csv lists demo + tick for each kill, which identifies the player to
anyone with the same (public) demo. Use it locally to review clips; publish aggregates only.

clean() and make_X() are also used by synthetic_eval.py, so both scripts apply exactly
the same filters and the same feature transform.
"""
import sys
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CLASSES = {
    "sniper": {"awp", "ssg08", "scar20", "g3sg1"},
    "rifle": {"ak47", "m4a1", "m4a1_silencer", "galilar", "famas", "aug", "sg556"},
    "smg": {"mp9", "mp7", "mp5sd", "mac10", "ump45", "p90", "bizon"},
    "pistol": {"glock", "usp_silencer", "hkp2000", "p250", "deagle",
               "fiveseven", "tec9", "cz75a", "elite", "revolver"},
}
WEAPON_TO_CLASS = {w: c for c, ws in CLASSES.items() for w in ws}

# Detector inputs. Degree errors are left out because they are distance-biased; the
# range-independent unit errors are used instead. distance_units is context rather than
# aim behaviour; it is kept so results stay comparable with earlier runs.
MODEL_FEATURES = ["peak_speed_dps", "mean_speed_dps", "speed_at_kill_dps", "peak_accel",
                  "settled_ticks", "distance_units",
                  "err_units_at_kill", "err_units_mean_last10", "settled_ticks_units"]
LOG_FEATURES = ["peak_speed_dps", "mean_speed_dps", "speed_at_kill_dps", "peak_accel"]


def load(path=None):
    if path is None:
        files = ["kill_features.csv"] + sorted(glob.glob("kill_features_*.csv"))
        path = max((f for f in files if os.path.exists(f)), key=os.path.getmtime)
    print(f"Loading {path}")
    return pd.read_csv(path)


def clean(df, verbose=True):
    """Keep only gun kills where aim is the story. Returns a new frame with a 'wclass' column."""
    say = print if verbose else (lambda *a, **k: None)
    df = df.copy()
    df["weapon"] = df["weapon"].astype(str).str.replace("weapon_", "", regex=False)
    if verbose:
        print("\nWeapon counts (top 25):")
        print(df["weapon"].value_counts().head(25).to_string())

    df["wclass"] = df["weapon"].map(WEAPON_TO_CLASS)
    dropped = df["wclass"].isna().sum()
    df = df.dropna(subset=["wclass"]).copy()
    say(f"\nDropped {dropped} non-gun / unclassified kills, kept {len(df)}")

    # Remove clearly broken rows (victim far outside view, e.g. blind/wallbang/glitch)
    before = len(df)
    df = df[df["err_at_kill_deg"] <= 20].copy()
    say(f"Dropped {before - len(df)} rows with aim error > 20 degrees")

    # Drop kills where aim isn't the story: wallbangs, through smoke, blind attacker
    flag_cols = [c for c in ("penetrated", "thrusmoke", "attackerblind") if c in df.columns]
    if flag_cols:
        before = len(df)
        df = df[~df[flag_cols].fillna(False).astype(bool).any(axis=1)].copy()
        say(f"Dropped {before - len(df)} wallbang / through-smoke / blind kills")

    # Point-blank kills inflate angular error (tiny offsets = huge angles)
    before = len(df)
    df = df[df["distance_units"] >= 150].copy()
    say(f"Dropped {before - len(df)} point-blank kills (< 150 units)")

    # Only keep kills where we found the shot burst (aim is measured at the first shot)
    if "has_shot_data" in df.columns:
        before = len(df)
        df = df[df["has_shot_data"].astype(bool)].copy()
        say(f"Dropped {before - len(df)} kills with no matching shot data")

    # Victims that did not move for 5 s (AFK / frozen / planting / defusing) say little about aim
    if "victim_moved_5s" in df.columns:
        before = len(df)
        df = df[df["victim_moved_5s"].fillna(1e9) >= 5].copy()
        say(f"Dropped {before - len(df)} kills on victims stationary for 5 s")
    return df


def make_X(df):
    """Detector input matrix (DataFrame, NaNs left in place; callers fill them)."""
    feats = [f for f in MODEL_FEATURES if f in df.columns]  # tolerate older CSVs
    X = df[feats].copy()
    for f in LOG_FEATURES:
        if f in X.columns:
            X[f] = np.log10(X[f] + 1)
    return X


def main():
    df = clean(load(sys.argv[1] if len(sys.argv) > 1 else None))

    print("\nKill counts by class and source:")
    print(df.groupby(["wclass", "source"]).size().unstack(fill_value=0).to_string())

    cols = ["peak_speed_dps", "err_at_kill_deg", "err_units_at_kill",
            "settled_ticks", "settled_ticks_units", "distance_units",
            "burst_shots", "victim_speed_ups"]
    cols = [c for c in cols if c in df.columns]  # tolerate older CSVs
    print("\nMedians by class and source:")
    print(df.groupby(["wclass", "source"])[cols].median().round(2).to_string())

    # Plots: one figure per feature, one panel per weapon class
    for feat, log in [("peak_speed_dps", True), ("err_at_kill_deg", False),
                      ("settled_ticks", False)]:
        classes = [c for c in CLASSES if (df["wclass"] == c).sum() > 30]
        if not classes:
            continue
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.2))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, classes):
            sub = df[df["wclass"] == c]
            vals = np.log10(sub[feat] + 1) if log else sub[feat]
            bins = np.histogram_bin_edges(vals, bins=30)
            for src, col in [("own", "tab:blue"), ("pro", "tab:orange")]:
                v = vals[sub["source"] == src]
                if len(v):
                    ax.hist(v, bins=bins, alpha=0.55, density=True, label=src, color=col)
            ax.set_title(c)
            ax.set_xlabel(f"log10({feat}+1)" if log else feat)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(f"plots_{feat}.png", dpi=130)
        plt.close(fig)
    print("\nSaved plots_*.png")

    # Anomaly detection per class (unsupervised, fitted on own + pro together)
    out = []
    for c in CLASSES:
        sub = df[df["wclass"] == c].copy()
        if len(sub) < 100:
            print(f"Skipping {c}: only {len(sub)} kills")
            continue
        X = make_X(sub)
        X = StandardScaler().fit_transform(X.fillna(X.median()))
        model = IsolationForest(n_estimators=300, contamination=0.02, random_state=0)
        model.fit(X)
        sub["anomaly_score"] = -model.score_samples(X)  # higher = more unusual
        sub["flagged"] = model.predict(X) == -1
        out.append(sub)

    if out:
        res = pd.concat(out)
        print("\nShare flagged as unusual, by class and source (expect similar if both groups look alike):")
        print(res.groupby(["wclass", "source"])["flagged"].mean().unstack().round(3).to_string())
        base = res["anchor_tick"] if "anchor_tick" in res.columns else res["tick"]
        res["goto_tick"] = (base - 192).clip(lower=0)  # ~3s before the first shot
        res.sort_values("anomaly_score", ascending=False).to_csv("anomalies.csv", index=False)
        print("Saved anomalies.csv (sorted, most unusual first). Local review only: "
              "demo + tick identifies the player, so do not publish its rows.")


if __name__ == "__main__":
    main()
