"""
train_model.py — fits and persists the heat anomaly detector.

"""

import glob
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

WINDOW = 21          # trailing days for seasonal baseline
N_COMPONENTS = 4
THRESHOLD_PCT = 95   # score above this percentile of training = anomaly
SEED = 42

# Outputs go next to this script, wherever it lives.
HERE = os.path.dirname(os.path.abspath(__file__))


def _search_roots():
    """This script's dir, the cwd, and their parents up to 3 levels."""
    roots = []
    for start in (HERE, os.getcwd()):
        d = start
        for _ in range(4):
            if d not in roots:
                roots.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return roots


def find_csv(stem):
    """Find a CSV by stem anywhere near the script or cwd.

    Tolerates '(1)', '__1_', spaces, and any containing folder, so it
    doesn't matter which directory you run from.
    """
    seen = []
    for root in _search_roots():
        for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            if ".git" in p:
                continue
            seen.append(p)
            if os.path.basename(p).lower().startswith(stem):
                return p
    raise FileNotFoundError(
        f"No CSV starting with '{stem}' found.\n"
        f"Searched: {_search_roots()}\n"
        f"CSVs seen: {sorted({os.path.basename(p) for p in seen}) or 'none'}"
    )


def load_data():
    heat = pd.read_csv(find_csv("heat_history"))
    env = pd.read_csv(find_csv("env_params"))
    heat["date"] = pd.to_datetime(heat["date"])
    env["date"] = pd.to_datetime(env["date"])
    # Both files are unsorted and in different orders — merge on date, never by index.
    df = heat.merge(env, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return df


def main():
    df = load_data()
    # max_temp/min_temp correlate 0.9999 with mean_temp (AOI aggregate, not a
    # spatial panel), so they add no information. Drop them.
    features = [c for c in df.columns if c not in ("date", "max_temp", "min_temp")]
    raw = df[features]

    # --- trailing seasonal baseline -------------------------------------
    # shift(1) so day t's baseline uses only days t-21..t-1, never day t itself.
    baseline = raw.rolling(WINDOW, min_periods=WINDOW).median().shift(1)
    resid = raw - baseline
    valid = resid.notna().all(axis=1)
    print(f"Rows: {len(df)} total, {valid.sum()} usable after {WINDOW}d warm-up")

    R = resid[valid].values

    # --- fit -------------------------------------------------------------
    scaler = StandardScaler().fit(R)
    Z = scaler.transform(R)
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED).fit(Z)

    def score(Zm):
        return np.sqrt(((Zm - pca.inverse_transform(pca.transform(Zm))) ** 2).sum(axis=1))

    train_scores = score(Z)
    threshold = float(np.percentile(train_scores, THRESHOLD_PCT))

    # per-feature robust spread, for attributing WHICH variable drove a flag
    med = np.median(R, axis=0)
    mad = np.median(np.abs(R - med), axis=0) * 1.4826
    mad = np.where(mad == 0, 1e-9, mad)

    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"Threshold (p{THRESHOLD_PCT}): {threshold:.3f}")

    # --- validation: synthetic anomaly injection -------------------------
    # We have no labels, so we manufacture some. Inject a known perturbation
    # into copies of real days and measure how often the detector catches it.
    # This is the honest answer to "how do you know it works?"
    rng = np.random.default_rng(SEED)
    results = {}
    for mag in (1.0, 2.0, 3.0, 4.0):
        idx = rng.choice(len(R), size=200, replace=True)
        corrupted = R[idx].copy()
        # perturb one randomly chosen feature by `mag` robust deviations
        cols = rng.integers(0, R.shape[1], size=len(corrupted))
        signs = rng.choice([-1, 1], size=len(corrupted))
        corrupted[np.arange(len(corrupted)), cols] += signs * mag * mad[cols]
        detected = (score(scaler.transform(corrupted)) > threshold).mean()
        results[f"{mag:.0f}_sigma"] = round(float(detected), 3)
        print(f"  detection @ {mag:.0f}x robust-sigma: {detected:.1%}")

    false_positive_rate = round(float((train_scores > threshold).mean()), 3)
    print(f"  false positive rate on clean data: {false_positive_rate:.1%}")

    # --- persist ----------------------------------------------------------
    # The buffer is essential: scoring a NEW day needs the trailing median of
    # the previous 21 days, so the model must carry that history with it.
    artifact = {
        "scaler": scaler,
        "pca": pca,
        "threshold": threshold,
        "features": features,
        "window": WINDOW,
        "mad": mad,
        "median": med,
        "buffer": raw.tail(WINDOW).values,          # last 21 days, raw
        "buffer_last_date": str(df["date"].iloc[-1].date()),
        "train_scores": train_scores,
        "validation": results,
        "false_positive_rate": false_positive_rate,
        "variance_explained": float(pca.explained_variance_ratio_.sum()),
        "n_train": int(valid.sum()),
    }
    joblib.dump(artifact, os.path.join(HERE, "model.joblib"))

    # scored history for the dashboard's overview tab
    out = df[valid].copy()
    out["anomaly_score"] = train_scores
    out["is_anomaly"] = train_scores > threshold
    rz = np.abs((R - med) / mad)
    out["driver"] = [features[i] for i in rz.argmax(axis=1)]
    out["driver_z"] = rz.max(axis=1)
    out.drop(columns=["max_temp", "min_temp"]).to_csv(os.path.join(HERE, "scored_history.csv"), index=False)

    print(f"\nWrote model.joblib and scored_history.csv to {HERE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()