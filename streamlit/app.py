"""
app.py — Streamlit dashboard for the heat anomaly detector.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Sentinel", page_icon="🔥", layout="wide")
 
# Artifacts may live next to this file or in a sibling folder (e.g. the
# trainer writes them beside train_model.py). Search rather than assume.
HERE = os.path.dirname(os.path.abspath(__file__))
 
 
def _search_roots():
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
 
 
def find_artifact(name):
    for root in _search_roots():
        for p in glob.glob(os.path.join(root, "**", name), recursive=True):
            if ".git" not in p:
                return p
    return None
 
 
@st.cache_resource
def load_model():
    path = find_artifact("model.joblib")
    if path is None:
        st.error(
            "**model.joblib not found.**\n\n"
            "Run `python train_model.py` first, then reload this page.\n\n"
            f"Searched: {_search_roots()}"
        )
        st.stop()
    return joblib.load(path)
 
 
@st.cache_data
def load_history():
    path = find_artifact("scored_history.csv")
    if path is None:
        st.error("**scored_history.csv not found.** Run `python train_model.py` first.")
        st.stop()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
 
 
M = load_model()
hist = load_history()
FEATURES = M["features"]
 
 
def score_day(values: dict):
    """Score one new day. Returns (score, is_anomaly, driver, driver_z)."""
    x = np.array([values[f] for f in FEATURES], dtype=float)
    base = np.median(M["buffer"], axis=0)      # trailing seasonal baseline
    r = (x - base).reshape(1, -1)
    z = M["scaler"].transform(r)
    recon = M["pca"].inverse_transform(M["pca"].transform(z))
    s = float(np.sqrt(((z - recon) ** 2).sum()))
    rz = np.abs((r.ravel() - M["median"]) / M["mad"])
    i = int(rz.argmax())
    return s, s > M["threshold"], FEATURES[i], float(rz[i])
 
 
GLOBE_HTML = """
<div style="display:flex;justify-content:center;align-items:center;height:210px;">
<svg viewBox="0 0 200 200" width="200" height="200"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="sphere" cx="35%" cy="30%">
      <stop offset="0%"  stop-color="#ff8a65"/>
      <stop offset="55%" stop-color="#d93025"/>
      <stop offset="100%" stop-color="#5c1008"/>
    </radialGradient>
    <radialGradient id="glow" cx="50%" cy="50%">
      <stop offset="70%" stop-color="#d93025" stop-opacity="0"/>
      <stop offset="100%" stop-color="#d93025" stop-opacity="0.35"/>
    </radialGradient>
    <clipPath id="ball"><circle cx="100" cy="100" r="72"/></clipPath>
  </defs>
 
  <circle cx="100" cy="100" r="88" fill="url(#glow)"/>
  <circle cx="100" cy="100" r="72" fill="url(#sphere)"/>
 
  <g clip-path="url(#ball)" fill="none" stroke="#ffe0d6"
     stroke-opacity="0.45" stroke-width="1">
    <ellipse cx="100" cy="100" rx="72" ry="24"/>
    <ellipse cx="100" cy="100" rx="72" ry="48"/>
    <ellipse cx="100" cy="100" rx="72" ry="66"/>
    <line x1="28" y1="100" x2="172" y2="100"/>
 
    <ellipse cx="100" cy="100" ry="72" rx="72">
      <animate attributeName="rx" values="72;0;72" dur="9s"
               repeatCount="indefinite"/>
    </ellipse>
    <ellipse cx="100" cy="100" ry="72" rx="48">
      <animate attributeName="rx" values="48;0;72;0;48" dur="9s"
               repeatCount="indefinite"/>
    </ellipse>
    <ellipse cx="100" cy="100" ry="72" rx="24">
      <animate attributeName="rx" values="24;72;0;24" dur="9s"
               repeatCount="indefinite"/>
    </ellipse>
  </g>
 
  <circle cx="100" cy="100" r="72" fill="none"
          stroke="#ffb3a0" stroke-opacity="0.55" stroke-width="1.2"/>
 
  <!-- AOI marker -->
  <g transform="translate(74,84)">
    <circle r="4" fill="#fff8f0"/>
    <circle r="4" fill="none" stroke="#fff8f0" stroke-width="1.5">
      <animate attributeName="r" values="4;16" dur="2.4s"
               repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.9;0" dur="2.4s"
               repeatCount="indefinite"/>
    </circle>
  </g>
</svg>
</div>
"""
 
with st.sidebar:
    components.html(GLOBE_HTML, height=215)
    st.markdown("### Sentinel")
    st.caption("by Team Ember Index")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Score a New Day", "Model Validation"],
        label_visibility="collapsed",
    )
 
st.title("Sentinel")
st.markdown("###### Detecting heat anomalies in urban environments · by Ember Index")
 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training days", M["n_train"])
c2.metric("Variance explained", f"{M['variance_explained']:.1%}")
c3.metric("Detection @ 3σ", f"{M['validation']['3_sigma']:.0%}")
c4.metric("False positive rate", f"{M['false_positive_rate']:.1%}")
 
 
# ------------------------------------------------------------- page: Overview
if page == "Overview":
    st.subheader("120-day anomaly timeline")
    flagged = hist[hist.is_anomaly]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(hist.date, hist.mean_temp, color="#444", lw=1.2, label="Mean temp")
    ax1.scatter(flagged.date, flagged.mean_temp, s=95, facecolor="none",
                edgecolor="#d93025", lw=2, zorder=5, label="Flagged anomaly")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=.25)
 
    ax2.fill_between(hist.date, hist.anomaly_score, color="#d93025", alpha=.28)
    ax2.plot(hist.date, hist.anomaly_score, color="#d93025", lw=1)
    ax2.axhline(M["threshold"], color="#333", ls=":", lw=1.2,
                label=f"Threshold {M['threshold']:.2f}")
    ax2.set_ylabel("Anomaly score")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=.25)
    plt.tight_layout()
    st.pyplot(fig)
 
    st.subheader("Flagged days")
    st.dataframe(
        flagged[["date", "mean_temp", "anomaly_score", "driver", "driver_z"]]
        .sort_values("anomaly_score", ascending=False)
        .assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d"))
        .round(2),
        use_container_width=True, hide_index=True)
 
    st.info(
        "Flags cluster around mid-June to mid-July — the onset of the Arizona "
        "monsoon, when Phoenix shifts from dry extreme heat to humid heat. "
        "The model was given no seasonal labels; it recovers this transition "
        "from the data alone."
    )
 
# --------------------------------------------------- page: Score a New Day
elif page == "Score a New Day":
    st.subheader("Score an unseen day")
    st.caption(
        f"Baseline is the trailing median of the 21 days ending "
        f"{M['buffer_last_date']}. Adjust the inputs to test the detector."
    )
 
    defaults = np.median(M["buffer"], axis=0)
    vals, cols = {}, st.columns(3)
    for i, f in enumerate(FEATURES):
        lo, hi = float(M["buffer"][:, i].min()), float(M["buffer"][:, i].max())
        span = max(hi - lo, 1.0)
        vals[f] = cols[i % 3].slider(
            f.replace("_", " ").replace(":", " ").title(),
            round(lo - 2 * span, 1), round(hi + 2 * span, 1),
            round(float(defaults[i]), 1))
 
    s, is_anom, driver, dz = score_day(vals)
    st.divider()
    a, b = st.columns([1, 2])
    a.metric("Anomaly score", f"{s:.2f}", f"threshold {M['threshold']:.2f}")
    if is_anom:
        b.error(f"**ANOMALY** — driven by `{driver}` at {dz:.1f}× robust sigma "
                f"from its seasonal baseline.")
    else:
        b.success(f"**Normal** — within expected range. Largest deviation: "
                  f"`{driver}` at {dz:.1f}× robust sigma.")
 
    with st.expander("Score a batch from CSV"):
        st.caption(f"Upload a CSV with these columns: {', '.join(FEATURES)}")
        up = st.file_uploader("CSV", type="csv", label_visibility="collapsed")
        if up is not None:
            new = pd.read_csv(up)
            missing = set(FEATURES) - set(new.columns)
            if missing:
                st.error(f"Missing columns: {sorted(missing)}")
            else:
                rows = [score_day(r) for _, r in new[FEATURES].iterrows()]
                new["anomaly_score"] = [r[0] for r in rows]
                new["is_anomaly"] = [r[1] for r in rows]
                new["driver"] = [r[2] for r in rows]
                st.dataframe(new.round(2), use_container_width=True)
                st.metric("Flagged", f"{new.is_anomaly.sum()} / {len(new)}")
 
# ------------------------------------------------------ page: Model Validation
elif page == "Model Validation":
    st.subheader("How we validated without labels")
    st.markdown(
        "Unsupervised anomaly detection has no ground truth, so we manufacture "
        "it. We take real days, inject a perturbation of known size into one "
        "randomly chosen variable, and measure how often the detector catches "
        "it. Perturbation size is expressed in robust standard deviations "
        "(MAD-scaled), so it is comparable across variables."
    )
 
    v = M["validation"]
    xs = [1, 2, 3, 4]
    ys = [v[f"{m}_sigma"] for m in xs]
 
    fig2, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, [y * 100 for y in ys], "o-", color="#d93025", lw=2, ms=7)
    ax.axhline(M["false_positive_rate"] * 100, color="#666", ls="--", lw=1.2,
               label=f"False positive rate ({M['false_positive_rate']:.1%})")
    ax.set_xlabel("Injected anomaly size (robust σ)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_xticks(xs)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.25)
    plt.tight_layout()
    st.pyplot(fig2)
 
    st.dataframe(pd.DataFrame({
        "Injected size": [f"{m}× robust σ" for m in xs],
        "Detection rate": [f"{y:.1%}" for y in ys],
    }), use_container_width=True, hide_index=True)
 
    st.markdown(
        f"""
**Reading this:** at a **{M['false_positive_rate']:.1%}** false positive rate the
detector catches **{v['3_sigma']:.0%}** of 3σ anomalies and **{v['4_sigma']:.0%}**
of 4σ anomalies. Small 1σ deviations are mostly missed by design — flagging
those would bury operators in noise.
 
**Known limitations, stated plainly:**
- Single AOI aggregate, not a spatial panel. No per-building ranking. Tile-level
  data would be the natural next step.
- 120 days covers one spring-to-summer transition. A winter cold anomaly is
  outside anything the model has seen.
- The trailing 21-day baseline means the first three weeks of any new
  deployment are a warm-up period.
"""
    )