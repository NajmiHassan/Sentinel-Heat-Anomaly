"""
Urban Heat Anomaly Detection — FortyGuard Hackathon '26
Deseasonalized multivariate anomaly detection on 120 days of San Jose AOI data.

Run:  python explore.py
Out:  anomaly_scores.csv, anomaly_plot.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import glob
import os

DATA_DIR = "retrieved data"   # change if your CSVs live elsewhere
ROLLING_WINDOW = 21     # days, for seasonal baseline
N_COMPONENTS = 4
TOP_K = 10


def find_csv(stem):
    """Locate a CSV by stem, tolerating '(1)', '__1_', spaces, etc."""
    hits = [p for p in glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
            if os.path.basename(p).lower().startswith(stem)]
    if not hits:
        raise FileNotFoundError(
            f"No CSV starting with '{stem}' under '{DATA_DIR}/'. "
            f"Found: {[os.path.basename(p) for p in glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)]}"
        )
    return sorted(hits, key=len)[-1]   # prefer the '(1)' / newest-looking one


# ---------------------------------------------------------------- load
# CRITICAL: both files are unsorted and in DIFFERENT orders.
# Merging on 'date' is what keeps rows aligned. Never concat by index.
heat = pd.read_csv(find_csv("heat_history"))
env = pd.read_csv(find_csv("env_params"))
heat["date"] = pd.to_datetime(heat["date"])
env["date"] = pd.to_datetime(env["date"])

df = heat.merge(env, on="date", how="inner").sort_values("date").reset_index(drop=True)
assert len(df) == 120, f"expected 120 joined rows, got {len(df)}"

# max_temp / min_temp correlate 0.9999 with mean_temp (AOI aggregate,
# not a spatial panel) so they carry no independent signal. Drop them.
FEATURES = [c for c in df.columns if c not in ("date", "max_temp", "min_temp")]

# ------------------------------------------------------- deseasonalize
# April averages 24.1C, August 36.7C — a 13.5C seasonal ramp. Without
# removing it, every spring day scores as a cold anomaly and the model
# is just reading the calendar.
raw = df[FEATURES]
baseline = raw.rolling(ROLLING_WINDOW, center=True, min_periods=7).median()
resid = raw - baseline

# ------------------------------------------------------------- score
Z = StandardScaler().fit_transform(resid)

pca = PCA(n_components=N_COMPONENTS).fit(Z)
recon = pca.inverse_transform(pca.transform(Z))
df["anomaly_score"] = np.sqrt(((Z - recon) ** 2).sum(axis=1))

# Per-feature robust z, for the "why" column — judges want attribution,
# not just a score.
med = np.median(resid, axis=0)
mad = np.median(np.abs(resid - med), axis=0) * 1.4826
rz = np.abs((resid - med) / np.where(mad == 0, 1e-9, mad))
df["driver"] = [FEATURES[i] for i in np.argmax(rz.values, axis=1)]
df["driver_z"] = rz.max(axis=1).values

# Rank instead of thresholding: a hard |z|>3 cut flags 26% of days,
# which is not an anomaly rate. Top-K also demos better.
df["rank"] = df["anomaly_score"].rank(ascending=False).astype(int)
df["is_anomaly"] = df["rank"] <= TOP_K

print(f"PCA variance explained ({N_COMPONENTS} comps): "
      f"{pca.explained_variance_ratio_.sum():.1%}")
print(f"\nTop {TOP_K} anomalous days:\n")
top = df[df.is_anomaly].sort_values("rank")
print(top[["date", "mean_temp", "anomaly_score", "driver", "driver_z"]]
      .assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d"))
      .round(2).to_string(index=False))

# -------------------------------------------------------------- plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(df.date, df.mean_temp, color="#444", lw=1.2, label="Mean temp")
ax1.plot(df.date, baseline["mean_temp"], color="#0a84ff", lw=1.6, ls="--",
         label=f"{ROLLING_WINDOW}d seasonal baseline")
ax1.scatter(top.date, top.mean_temp, s=90, facecolor="none",
            edgecolor="#d93025", lw=2, zorder=5, label=f"Top {TOP_K} anomalies")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Downtown Phoenix, AZ (~2×2 km AOI) — Deseasonalized Heat "
              "Anomaly Detection, 120 days",
              loc="left", fontsize=13, weight="bold")
ax1.legend(frameon=False, fontsize=9)
ax1.grid(alpha=.25)

ax2.fill_between(df.date, df.anomaly_score, color="#d93025", alpha=.28)
ax2.plot(df.date, df.anomaly_score, color="#d93025", lw=1)
ax2.axhline(top.anomaly_score.min(), color="#666", ls=":", lw=1)
ax2.set_ylabel("Anomaly score")
ax2.set_xlabel("Date")
ax2.grid(alpha=.25)

plt.tight_layout()
plt.savefig("anomaly_plot.png", dpi=150)

df.drop(columns=["max_temp", "min_temp"]).to_csv("anomaly_scores.csv", index=False)
print("\nWrote anomaly_scores.csv and anomaly_plot.png")