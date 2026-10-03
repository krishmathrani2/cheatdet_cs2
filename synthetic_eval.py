"""
Synthetic-cheat evaluation of the aim-anomaly detector.

Idea: take REAL engagements from your demos, edit the attacker's view-angle trace to
simulate aim assistance at different strengths, and measure how often a detector that
was trained only on real, unedited kills flags the edited version.

    python features.py --windows      # once: saves windows.pkl
    python synthetic_eval.py          # run the evaluation
    python synthetic_eval.py --fast   # fewer bootstrap / scarcity repeats
    python synthetic_eval.py --group-by demo   # old behaviour: each demo FILE is a CV group

Outputs:
    synthetic_results.csv                   metrics per target / cheat model / strength / class
    synthetic_detection.png                 detection vs strength, cheat aims at the measurement point
    synthetic_detection_realistic.png       same, cheat aims at the lagged head (see TARGETS)
    synthetic_results_player_disjoint.csv   sensitivity check: no test player in training
    scarcity_results.csv, scarcity.png      detection vs number of training matches

HOW TO READ AUC HERE: 0.5 = the edited kills look exactly as unusual as the real ones.
Above 0.5 = edited kills look MORE unusual. BELOW 0.5 = edited kills look MORE NORMAL
than real ones (the detector is two-sided: it also flags unusually bad aim).

CONTROLS
  * strength 0, no jitter: the edited trace is the real trace bit-for-bit. This only
    checks that real and edited kills go through the same code path; it is asserted
    at runtime. It cannot detect leakage or unrealistic injection.
  * "sham": same timing and same per-tick edit size as "smooth", but the edit rotates
    the aim error around the target instead of shrinking it, so every accuracy feature
    is unchanged. Detection of the sham = detection of "the trace was edited" rather
    than "aim got better". Read cheat results against it.
  * jitter: per-tick Gaussian noise (NOISES) on the last 30 ticks. "jitter_only" rows
    show what the jitter alone does. auc_vs_jittered_real compares an edited+jittered
    kill against a real kill with the same kind of jitter, so the jitter's own effect
    cancels and only the cheat remains.

IMPORTANT LIMITS (say these in any write-up):
  * The "cheats" are my own simple simulations (a blend of the real trace toward an
    aim point). Real cheats differ. These numbers say how detectable THESE signatures
    are, not how well the detector would catch real cheating.
  * The edits are open-loop: the real player reacted to what they saw WITHOUT the cheat.
  * The detector is unsupervised and trained on unlabelled matches. Some real kills in
    the training data could come from cheaters; there are no labels to check.
  * Evaluation is grouped by MATCH: parts of one map (p1/p2) and maps of one series
    (m1/m2) share players, so they are one group. No match is in both train and test,
    and confidence intervals resample whole matches. The demo recorder appears in every
    own demo, so the player-disjoint file shows how much that player overlap matters.
"""
import argparse
import re
import pickle
import zlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import features as F
try:
    import analyze as AN
except ImportError:          # you may have saved it as analysis.py
    import analysis as AN

# Fraction of the remaining aim error removed. Strength 0 is a pipeline check (see top).
STRENGTHS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
# Extra per-tick jitter (degrees) added over the last 30 ticks, independent of strength.
NOISES = [0.0, 0.1]
CHEATS = {
    "snap":   dict(ramp=2,  onset=(3, 10)),    # fast lock-on in the last few ticks before the shot
    "smooth": dict(ramp=15, onset=(20, 40)),   # slow, humanised pull toward the target
    "assist": dict(radius=6.0),                # magnetism: pulls only while already close (deg)
    "sham":   dict(ramp=15, onset=(20, 40)),   # CONTROL: smooth's timing, no accuracy gain
}
# Where the simulated cheat aims.
#   "reference": the exact point the error features measure against (victim upper body at
#                the same server tick). At strength 1 the measured error becomes exactly 0,
#                which real aim almost never shows, so this is an optimistic upper bound.
#   "realistic": the victim's head where the victim was a few ticks EARLIER, which is
#                roughly what a client-side cheat sees. Constants below come from the demos.
TARGETS = ["reference", "realistic"]
HEAD_AIM = 62.0     # first-shot headshots on standing victims line up ~62 units above the feet
CROUCH_DROP = 18.0  # eye / head height drop when fully crouched (64 -> 46)
LAG_BASE = 3        # view delay in ticks at 0 ping (LAN pro demos fit best at 2-3 ticks)
TICK_MS = 1000.0 / 64
FEATURE_COLS = ["peak_speed_dps", "mean_speed_dps", "speed_at_kill_dps", "peak_accel",
                "err_at_kill_deg", "err_mean_last10_deg", "err_min_deg", "settled_ticks",
                "err_units_at_kill", "err_units_mean_last10", "settled_ticks_units",
                "distance_units"]
ACCURACY_COLS = ["err_at_kill_deg", "err_mean_last10_deg", "err_min_deg",
                 "err_units_at_kill", "err_units_mean_last10"]
JITTER_ONLY = "jitter_only"


# ----------------------------------------------------------------------------- data

def match_group(demo: str) -> str:
    """One CV group per match. 'aurora-vs-9z-m2-mirage-p1' -> 'aurora-vs-9z'.
    Matchmaking demos (match730_...) are each their own match."""
    m = re.match(r"^(.+?-vs-.+?)-m\d+", str(demo))
    return m.group(1) if m else str(demo)


def window_feats(w, yaw=None, pitch=None):
    return F.window_features(w["t"], w["yaw"] if yaw is None else yaw,
                             w["pitch"] if pitch is None else pitch,
                             w["ax"], w["ay"], w["az"], w["vx"], w["vy"], w["vz"])


def build_real_table(windows, group_by="match"):
    """Recompute features from the raw windows (so real and injected rows come from the
    same code path), then apply the same cleaning filters as analysis.py."""
    rows = []
    for i, w in enumerate(windows):
        f = window_feats(w)
        if f is None:
            continue
        r = dict(w["meta"])
        r.update(f)
        r["win_id"] = i
        rows.append(r)
    df = pd.DataFrame(rows)
    df = AN.clean(df, verbose=False).reset_index(drop=True)
    df["group"] = df["demo"] if group_by == "demo" else df["demo"].map(match_group)
    return df


# ------------------------------------------------------------------------ injection

def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if np.isfinite(x) else 0.0


def cheat_target(w, target):
    """(yaw, pitch) the simulated cheat steers toward, per tick of the window."""
    if target == "reference":
        ty, tp, _ = F.ideal_angles(w["ax"], w["ay"], w["az"], w["vx"], w["vy"], w["vz"])
        return ty, tp
    m = w.get("meta", {})
    n = len(w["t"])
    lag = int(round(LAG_BASE + _num(m.get("attacker_ping_ms")) / TICK_MS))
    idx = np.clip(np.arange(n) - lag, 0, n - 1)
    ty, tp, _ = F.ideal_angles(w["ax"], w["ay"], w["az"], w["vx"][idx], w["vy"][idx], w["vz"][idx],
                               eye=F.EYE_HEIGHT - CROUCH_DROP * _num(m.get("attacker_duck")),
                               aim=HEAD_AIM - CROUCH_DROP * _num(m.get("victim_duck")))
    return ty, tp


def inject(w, cheat, s, noise, rng, target="reference"):
    """Return (yaw, pitch) of a simulated assisted version of the window.
    modified = real + weight * (aim point - real), with a cheat-specific weight profile.
    Ticks that are not edited are returned bit-for-bit unchanged."""
    yaw, pitch = w["yaw"], w["pitch"]
    n = len(yaw)
    # The sham must keep the MEASURED error, so it always rotates around the reference point.
    tgt_yaw, tgt_pitch = cheat_target(w, "reference" if cheat == "sham" else target)
    d_yaw = F.wrap(tgt_yaw - yaw)
    d_pitch = tgt_pitch - pitch
    if cheat == JITTER_ONLY:
        weight = np.zeros(n)
    elif cheat == "assist":
        err = np.hypot(d_yaw, d_pitch)
        weight = s * np.clip(1.0 - err / CHEATS["assist"]["radius"], 0.0, 1.0)
    else:
        p = CHEATS[cheat]
        onset = int(rng.integers(p["onset"][0], p["onset"][1] + 1))   # ticks before the shot
        start = max(n - 1 - onset, 0)
        weight = s * np.clip((np.arange(n) - start) / p["ramp"], 0.0, 1.0)
    weight = np.where(np.isfinite(weight), weight, 0.0)   # missing positions: leave the tick alone

    if cheat == "sham":
        # Rotate the error vector (real - target) by phi around the target. The error size is
        # unchanged, and the chord length 2*|e|*sin(phi/2) equals weight*|e|, the size of the
        # smooth cheat's edit at the same tick.
        phi = 2.0 * np.arcsin(np.clip(weight / 2.0, 0.0, 1.0)) * (1.0 if rng.random() < 0.5 else -1.0)
        ex, ey = -d_yaw, -d_pitch
        c, sn = np.cos(phi), np.sin(phi)
        dy2 = (c * ex - sn * ey) - ex
        dp2 = (sn * ex + c * ey) - ey
    else:
        dy2 = np.where(weight != 0, weight * d_yaw, 0.0)
        dp2 = np.where(weight != 0, weight * d_pitch, 0.0)
    changed = weight != 0
    if noise > 0:
        active = np.arange(n) >= n - 30      # jitter is independent of the cheat's strength
        dy2 = dy2 + active * rng.normal(0.0, noise, n)
        dp2 = dp2 + active * rng.normal(0.0, noise, n)
        changed = changed | active
    if not changed.any():
        return yaw, pitch
    return np.where(changed, F.wrap(yaw + dy2), yaw), np.where(changed, pitch + dp2, pitch)


def injected_table(windows, df, cheat, s, noise, target="reference"):
    """Feature table for the injected version of every row in df (same index)."""
    tag = f"{cheat}|{s}|{noise}" + ("" if target == "reference" else f"|{target}")
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    recs = []
    for _, win_id in df["win_id"].items():
        w = windows[int(win_id)]
        yaw2, pitch2 = inject(w, cheat, s, noise, rng, target)
        f = window_feats(w, yaw2, pitch2)
        recs.append(f if f is not None else {k: np.nan for k in FEATURE_COLS})
    new = df.copy()
    new[FEATURE_COLS] = pd.DataFrame(recs, index=df.index)[FEATURE_COLS].to_numpy(dtype=float)
    return new


def all_keys():
    keys = []
    for target in TARGETS:
        for cheat in CHEATS:
            if cheat == "sham" and target != "reference":
                continue
            for s in STRENGTHS:
                for noise in NOISES:
                    keys.append((target, cheat, s, noise))
    keys += [("reference", JITTER_ONLY, 0.0, j) for j in NOISES if j > 0]
    return keys


# ----------------------------------------------------------------------- detector

class Detector:
    """One Isolation Forest per weapon class, trained on real kills only.
    score() returns a z-score against the training score distribution, so scores from
    different weapon classes can be pooled. threshold() is the 95th percentile of the
    training z-scores: a cut-off you could fix BEFORE seeing any test data."""

    def fit(self, train, seed=0):
        self.m = {}
        for c, g in train.groupby("wclass"):
            if len(g) < 40:
                continue
            X = AN.make_X(g)
            med = X.median()
            Xf = X.fillna(med)
            sc = StandardScaler().fit(Xf)
            iso = IsolationForest(n_estimators=200, random_state=seed).fit(sc.transform(Xf))
            tr = -iso.score_samples(sc.transform(Xf))
            mu, sd = float(tr.mean()), float(tr.std() + 1e-9)
            self.m[c] = (med, sc, iso, mu, sd, float(np.quantile((tr - mu) / sd, 0.95)))
        return self

    def score(self, df):
        out = np.full(len(df), np.nan)
        for c, (med, sc, iso, mu, sd, _) in self.m.items():
            mask = (df["wclass"] == c).to_numpy()
            if not mask.any():
                continue
            X = AN.make_X(df[mask]).fillna(med)
            out[mask] = (-iso.score_samples(sc.transform(X)) - mu) / sd
        return out

    def threshold(self, df):
        out = np.full(len(df), np.nan)
        for c, model in self.m.items():
            out[(df["wclass"] == c).to_numpy()] = model[5]
        return out


def metrics(neg, pos, thr=None):
    """AUC and true-positive rate at a 5% false-positive rate. The 5% point is read off
    the ROC curve of the held-out real kills themselves (an oracle threshold). With
    `thr` (thresholds fixed on TRAINING data) it also returns the false- and true-positive
    rates you would actually get with a cut-off chosen in advance."""
    ok = ~np.isnan(neg) & ~np.isnan(pos)
    if ok.sum() < 20:
        return None
    n, p = neg[ok], pos[ok]
    out = {
        "auc": roc_auc_score(np.r_[np.zeros(len(n)), np.ones(len(p))], np.r_[n, p]),
        "tpr": float((p > np.quantile(n, 0.95)).mean()),
        "n": int(ok.sum()),
    }
    if thr is not None:
        out["fpr_train_thr"] = float((n > thr[ok]).mean())
        out["tpr_train_thr"] = float((p > thr[ok]).mean())
    return out


def cluster_ci(neg, pos, groups, reps, seed=0):
    """95% interval by resampling whole matches (CV groups)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    aucs, tprs = [], []
    for _ in range(reps):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        m = metrics(neg[idx], pos[idx])
        if m:
            aucs.append(m["auc"])
            tprs.append(m["tpr"])
    if not aucs:
        return (np.nan, np.nan), (np.nan, np.nan)
    return tuple(np.percentile(aucs, [2.5, 97.5])), tuple(np.percentile(tprs, [2.5, 97.5]))


# ------------------------------------------------------------------- experiments

def cross_validate(df, inj, n_splits=5, seed=0, player_disjoint=False):
    """Grouped K-fold over matches. With player_disjoint, kills by any attacker who also
    appears in the test fold are removed from training as well."""
    groups = df["group"].to_numpy()
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    neg = np.full(len(df), np.nan)
    thr = np.full(len(df), np.nan)
    pos = {k: np.full(len(df), np.nan) for k in inj}
    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        train = df.iloc[tr_idx]
        if player_disjoint:
            train = train[~train["player"].isin(set(df["player"].iloc[te_idx]))]
        det = Detector().fit(train, seed)
        te = df.iloc[te_idx]
        neg[te_idx] = det.score(te)
        thr[te_idx] = det.threshold(te)
        for k, tbl in inj.items():
            pos[k][te_idx] = det.score(tbl.iloc[te_idx])
    return neg, pos, thr


def detection_results(df, neg, pos, thr, reps):
    groups = df["group"].to_numpy()
    classes = df["wclass"].to_numpy()
    rows = []
    for k, p in pos.items():
        target, cheat, s, noise = k
        m = metrics(neg, p, thr)
        if m is None:
            continue
        (alo, ahi), (tlo, thi) = cluster_ci(neg, p, groups, reps) if reps else ((np.nan,) * 2,) * 2
        row = dict(target=target, cheat=cheat, strength=s, noise=noise, scope="pooled", n=m["n"],
                   auc=m["auc"], auc_lo=alo, auc_hi=ahi, tpr_at_5pct_fpr=m["tpr"],
                   tpr_lo=tlo, tpr_hi=thi, fpr_at_train_threshold=m["fpr_train_thr"],
                   tpr_at_train_threshold=m["tpr_train_thr"])
        jk = ("reference", JITTER_ONLY, 0.0, noise)
        if noise > 0 and cheat != JITTER_ONLY and jk in pos:
            mj = metrics(pos[jk], p)
            if mj:
                row.update(auc_vs_jittered_real=mj["auc"], tpr_vs_jittered_real=mj["tpr"])
        rows.append(row)
        for c in sorted(set(classes)):
            mc = metrics(np.where(classes == c, neg, np.nan), np.where(classes == c, p, np.nan))
            if mc:
                rows.append(dict(target=target, cheat=cheat, strength=s, noise=noise, scope=c,
                                 n=mc["n"], auc=mc["auc"], tpr_at_5pct_fpr=mc["tpr"]))
    return pd.DataFrame(rows)


def plot_detection(res, target, path):
    pooled = res[(res["scope"] == "pooled") & (res["target"] == target)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"snap": "tab:red", "smooth": "tab:blue", "assist": "tab:green", "sham": "tab:gray"}
    for ax, (col, lo, hi, title, chance) in zip(axes, [
            ("auc", "auc_lo", "auc_hi", "AUC (held-out matches)", 0.5),
            ("tpr_at_5pct_fpr", "tpr_lo", "tpr_hi", "Detection rate at 5% false positives", 0.05)]):
        for cheat in CHEATS:
            for noise, ls in [(0.0, "-"), (NOISES[-1], "--")]:
                g = pooled[(pooled["cheat"] == cheat) & (pooled["noise"] == noise)].sort_values("strength")
                if g.empty or (cheat == "sham" and noise):
                    continue
                label = ("sham control" if cheat == "sham" else cheat) + (f" +{noise}° jitter" if noise else "")
                ax.plot(g["strength"], g[col], ls, marker="o", color=colors[cheat], label=label)
                if noise == 0.0:
                    ax.fill_between(g["strength"], g[lo], g[hi], color=colors[cheat], alpha=0.15)
        ax.axhline(chance, color="gray", lw=1, ls=":")
        ax.set_xlabel("simulated cheat strength (fraction of aim error removed)")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"cheat aims at: {target}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scarcity(df, inj, keys, sizes=(2, 4, 6, 8, 12, 16), reps=8, n_test=5, seed=0):
    """Detection as a function of how many groups (matches) the detector was trained on.
    Within one repeat every training size is scored on the SAME test kills (those every
    size could score), so the curve is not confounded by weapon classes dropping out
    when small training sets have too few kills of a class."""
    rng = np.random.default_rng(seed)
    groups = df["group"].to_numpy()
    ug = np.unique(groups)
    n_test = min(n_test, max(1, len(ug) // 4))
    rows = []
    for r in range(reps):
        perm = rng.permutation(ug)
        test_g, train_g = perm[:n_test], perm[n_test:]
        te = np.isin(groups, test_g)
        scored = {}
        for n in sizes:
            if n > len(train_g):
                continue
            tr = np.isin(groups, train_g[:n])
            det = Detector().fit(df[tr], seed + r)
            scored[n] = (int(tr.sum()), det.score(df[te]),
                         {k: det.score(inj[k][te]) for k in keys})
        if not scored:
            continue
        common = np.all([~np.isnan(v[1]) for v in scored.values()], axis=0)
        for n, (n_kills, neg, pos) in scored.items():
            for k in keys:
                m = metrics(np.where(common, neg, np.nan), np.where(common, pos[k], np.nan))
                if m:
                    rows.append(dict(rep=r, n_train_groups=n, n_train_kills=n_kills,
                                     n_test_kills_scored=m["n"], target=k[0], cheat=k[1],
                                     strength=k[2], auc=m["auc"], tpr_at_5pct_fpr=m["tpr"]))
    return pd.DataFrame(rows)


def plot_scarcity(sc, path="scarcity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (col, title) in zip(axes, [("auc", "AUC"), ("tpr_at_5pct_fpr", "Detection rate at 5% false positives")]):
        for (target, cheat, s), g in sc.groupby(["target", "cheat", "strength"]):
            a = g.groupby("n_train_groups")[col].agg(["mean", "std"]).fillna(0)
            ax.errorbar(a.index, a["mean"], yerr=a["std"], marker="o", capsize=3,
                        label=f"{cheat}, strength {s}, {target}")
        ax.set_xlabel("number of training groups (matches, or demos with --group-by demo)")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pivot(res, value, target, noise, cheats=None):
    p = res[(res["scope"] == "pooled") & (res["target"] == target) & (res["noise"] == noise)]
    if cheats:
        p = p[p["cheat"].isin(cheats)]
    return p.pivot(index="cheat", columns="strength", values=value).round(2).to_string()


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Synthetic-cheat evaluation of the aim-anomaly detector.")
    ap.add_argument("windows", nargs="?", default="windows.pkl")
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap and scarcity repeats")
    ap.add_argument("--group-by", choices=["match", "demo"], default="match",
                    help="CV / bootstrap unit (default: match; 'demo' = old behaviour)")
    a = ap.parse_args()
    fast, group_by, path = a.fast, a.group_by, a.windows
    with open(path, "rb") as fh:
        windows = pickle.load(fh)
    print(f"Loaded {len(windows)} windows from {path}")

    df = build_real_table(windows, group_by)
    print(f"{len(df)} eligible kills after cleaning, from {df['demo'].nunique()} demos "
          f"= {df['group'].nunique()} independent {group_by} groups for CV")
    print(df.groupby(["wclass", "source"]).size().unstack(fill_value=0).to_string())

    print("\nBuilding simulated-cheat versions of every eligible kill ...")
    inj = {k: injected_table(windows, df, k[1], k[2], k[3], k[0]) for k in all_keys()}

    # --- checks that the comparison is like-for-like
    real = df[FEATURE_COLS].to_numpy(dtype=float)
    for k, tbl in inj.items():
        if k[2] == 0.0 and k[3] == 0.0 and not np.array_equal(
                tbl[FEATURE_COLS].to_numpy(dtype=float), real, equal_nan=True):
            raise SystemExit(f"Negative control FAILED for {k}: strength 0 changed the features")
    print("Check: strength 0 without jitter reproduces every real kill's features exactly.")
    sham_dev = max(np.nanmax(np.abs(inj[("reference", "sham", s, 0.0)][ACCURACY_COLS].to_numpy(dtype=float)
                                    - df[ACCURACY_COLS].to_numpy(dtype=float))) for s in STRENGTHS)
    print(f"Check: the sham leaves every accuracy feature unchanged (largest difference {sham_dev:.1e}).")
    print("How often is the aim error at the first shot under 0.05 degrees?")
    print(f"   real kills {np.mean(df['err_at_kill_deg'] < 0.05):.1%}")
    for target in TARGETS:
        e = inj[(target, "smooth", 1.0, 0.0)]["err_at_kill_deg"]
        print(f"   smooth cheat, strength 1, aiming at {target:9s} {np.mean(e < 0.05):.1%}")

    print(f"\nCross-validating (grouped by {group_by}) ...")
    neg, pos, thr = cross_validate(df, inj)
    res = detection_results(df, neg, pos, thr, reps=60 if fast else 300)
    res.to_csv("synthetic_results.csv", index=False)
    plot_detection(res, "reference", "synthetic_detection.png")
    plot_detection(res, "realistic", "synthetic_detection_realistic.png")

    fpr = res.loc[res["scope"] == "pooled", "fpr_at_train_threshold"].iloc[0]
    print(f"\nWith a threshold fixed on training data (its 95th percentile), {fpr:.1%} of held-out "
          "real kills are flagged (5% would mean the threshold transfers to new matches).")
    for target in TARGETS:
        print(f"\n=== Cheat aims at: {target} ===")
        print("Pooled AUC, no jitter (sham = control, should stay near 0.5):")
        print(pivot(res, "auc", target, 0.0))
        print("Pooled detection rate at 5% false positives (ROC point), no jitter:")
        print(pivot(res, "tpr_at_5pct_fpr", target, 0.0))
        print("Pooled detection rate with the threshold fixed on training data, no jitter:")
        print(pivot(res, "tpr_at_train_threshold", target, 0.0))
        j = NOISES[-1]
        print(f"Pooled AUC with {j} deg jitter, against UNTOUCHED real kills:")
        print(pivot(res, "auc", target, j, cheats=[c for c in CHEATS if c != "sham"]))
        print(f"Pooled AUC with {j} deg jitter, against real kills with the SAME jitter "
              "(jitter's own effect cancels):")
        print(pivot(res, "auc_vs_jittered_real", target, j, cheats=[c for c in CHEATS if c != "sham"]))
    jo = res[(res["cheat"] == JITTER_ONLY) & (res["scope"] == "pooled")]
    for _, r in jo.iterrows():
        print(f"\nJitter alone ({r['noise']} deg, no cheat) vs untouched real: AUC {r['auc']:.2f} "
              f"[{r['auc_lo']:.2f}, {r['auc_hi']:.2f}]")

    print("\nSensitivity: player-disjoint CV (no attacker from the test fold in training) ...")
    keys0 = [k for k in inj if k[3] == 0.0]
    neg_p, pos_p, thr_p = cross_validate(df, {k: inj[k] for k in keys0}, player_disjoint=True)
    res_p = detection_results(df, neg_p, pos_p, thr_p, reps=0)
    res_p = res_p[res_p["scope"] == "pooled"]
    res_p.to_csv("synthetic_results_player_disjoint.csv", index=False)
    both = res_p.merge(res[(res["scope"] == "pooled") & (res["noise"] == 0.0)],
                       on=["target", "cheat", "strength", "noise"], suffixes=("_disjoint", "_main"))
    both = both[both["strength"].isin([0.5, 1.0])]
    print(both[["target", "cheat", "strength", "auc_main", "auc_disjoint"]].round(2).to_string(index=False))

    print("\nData-scarcity study ...")
    keys = [("reference", "smooth", 0.5, 0.0), ("reference", "snap", 0.5, 0.0),
            ("realistic", "smooth", 1.0, 0.0)]
    sc = scarcity(df, inj, keys, reps=3 if fast else 8)
    sc.to_csv("scarcity_results.csv", index=False)
    if len(sc):
        plot_scarcity(sc)
        print(sc.groupby(["target", "cheat", "strength", "n_train_groups"])["auc"]
              .agg(["mean", "std"]).round(3).to_string())
    print("\nSaved synthetic_results.csv, synthetic_detection*.png, "
          "synthetic_results_player_disjoint.csv, scarcity_results.csv, scarcity.png")


if __name__ == "__main__":
    main()
