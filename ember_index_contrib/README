# Sentinel: Urban Heat Anomaly Detection

**Team Ember Index** · FortyGuard Hackathon '26 · Track 05, Model Designing

An unsupervised anomaly detector for urban heat data. It learns how environmental
variables normally co-vary in a city, then flags days where that relationship
breaks, and names the variable responsible.

Built on the [FortyGuard tOS Enterprise API](https://www.fortyguard.com/) using
this quickstart repo as the data layer.

---

## The problem we chose

Track 05 offers three model types: forecasting, risk classification, and anomaly
detection. We profiled the data before choosing, and the data ruled out two of them.

| Approach | Why we rejected it |
|---|---|
| **Forecasting** | 120 daily observations, one series, no year-over-year baseline. The predictive features are same-day readings, so live forecasting would require forecasting the drivers first. |
| **Risk classification** | No labels exist. The only derivable label is "hours above 32°C", a deterministic function of temperature, which is also the input feature. The model would learn the threshold and nothing else. |
| **Anomaly detection** | Works unsupervised, needs no labels, and produces per-day attribution operators can act on. |

Choosing the model the data supports, rather than the most impressive-sounding
one, is the core engineering decision in this submission.

## Data

- **AOI:** ~2 km × 2 km over Downtown Phoenix, Arizona
- **Window:** 2026-04-20 to 2026-08-17 (120 consecutive days, no gaps, no nulls)
- **Features:** 11 environmental variables covering temperature, heat index,
  apparent temperature, wet bulb temperature, relative humidity, and air quality
  index (mean and max where available)

`max_temp` and `min_temp` correlate with `mean_temp` at 0.9999 (spread averages
0.21°C), confirming these are AOI aggregates rather than a spatial panel. Both
are dropped as uninformative.

**Data integrity note:** the two source CSVs arrive in different row orders. They
must be joined on `date`, never concatenated by index. Index alignment produces a
pipeline that runs successfully and is silently wrong.

## Method

**1. Deseasonalize.** Temperature rises 0.113°C/day across the window, roughly
13.5°C from April to August. Without removing this, every spring day scores as a
cold anomaly and the detector becomes a calendar. We subtract each feature's
**trailing** 21-day rolling median, shifted by one day.

The trailing window matters. A centered rolling median looks forward in time,
which is acceptable for retrospective analysis but makes live scoring impossible.
Trailing costs a 21-day warm-up (120 rows become 99 usable) and buys a model that
runs on tomorrow's data.

**2. Standardize.** `StandardScaler` places percentages and degrees Celsius on
comparable footing.

**3. PCA reconstruction error.** Four components capture **94.9%** of variance in
the deseasonalized residuals. PC1 loads positively on heat index, wet bulb, and
apparent temperature together, a humid-heat axis discovered from the data with no
supervision.

Each day is projected into that subspace and back out. Normal days reconstruct
almost exactly. Anomalous days violate the learned correlation structure, so
reconstruction fails and the error spikes. That error is the anomaly score.

**4. Attribution.** A MAD-scaled robust z-score per feature identifies which
variable drove each flag, so output reads "flagged: relative humidity at 4.2x
robust sigma" rather than an opaque number.

Threshold is the 95th percentile of training scores. 100 learned parameters total.

## Validation without labels

Unsupervised detection has no ground truth, so we manufacture it. We take real
days, inject a perturbation of known magnitude into one randomly selected
variable, and measure recovery. Magnitudes are in robust (MAD-scaled) standard
deviations, making them comparable across features.

| Injected anomaly | Detection rate |
|---|---|
| 1x robust sigma | 16.0% |
| 2x robust sigma | 63.5% |
| 3x robust sigma | 90.5% |
| 4x robust sigma | 97.0% |
| **False positive rate (clean data)** | **5.1%** |

At a 5.1% false positive rate the detector catches 90.5% of 3 sigma anomalies.
Small 1 sigma deviations are missed by design, since flagging those would bury
operators in noise.

Every number here regenerates deterministically from `train_model.py` (seed 42).

## What the model found

Flagged days cluster around the mid-June to mid-July period. Mean relative
humidity in the AOI shifts from **16.6%** before June 15 to **26.5%** after,
consistent with the Arizona monsoon transition, when Phoenix moves from dry
extreme heat toward humid heat.

The model received no seasonal labels and no calendar features. It identified the
period where the relationship between heat variables changes character, which
coincides with a known meteorological transition. That the flags align with real
regional climatology is evidence the detector responds to physical structure
rather than noise.

Independently, the AOI's monthly humidity profile (15.0% June, 30.0% July, 30.1%
August) tracks published Phoenix climate normals (16%, 27%, 32%), a useful sanity
check on the data pipeline.

## Limitations

Stated plainly, because a detector whose failure modes are unknown is not
deployable:

- **Single AOI aggregate, not a spatial panel.** No per-building or per-tile
  ranking. Tile-level data is the natural next step.
- **One seasonal cycle.** 120 days covers a single spring-to-summer transition. A
  winter cold anomaly falls outside anything the model has observed.
- **21-day warm-up.** Any new deployment needs three weeks of history before the
  trailing baseline is defined.
- **Detection is weak below 2 sigma**, by design.
- The AOI runs warm relative to 1991-2020 climate normals. We do **not** claim
  this as an urban heat island measurement. Isolating that effect would require a
  same-year reference station, which we do not have.

## Running it

```bash
pip install -r requirements.txt
python ember_index_contrib/train_model.py    # writes model.joblib + scored_history.csv
streamlit run streamlit/app.py               # launches Sentinel
```

`train_model.py` locates the CSVs by searching nearby directories, so it runs
from any working directory.

## Dashboard

Three views:

- **Overview:** 120-day timeline with flagged days and a ranked anomaly table
  showing the driving variable for each
- **Score a New Day:** load the saved model and score unseen data, via sliders or
  CSV upload; returns score, verdict, and attribution
- **Model Validation:** the injection-recovery curve above

## Files

```
ember_index_contrib/
├── train_model.py       fits and persists the detector
├── explore.py           exploratory profiling pass
├── model.joblib         scaler, PCA, threshold, 21-day buffer
└── scored_history.csv   scored 120-day history

streamlit/
└── app.py               Sentinel dashboard
```

The persisted model carries a 21-day raw history buffer, which is what allows a
single new observation to be scored against its trailing seasonal baseline.

## Credits

Forked from FortyGuard's `temperature-api-quickstart`. The `fortyguard/`,
`notebooks/`, `data/`, and `docs/` directories are upstream. Our contribution is
`ember_index_contrib/` and `streamlit/`.

Data retrieved via the FortyGuard tOS Enterprise API.

---

*Built with AI assistance (Claude) for code review, model design, and
documentation. All modeling decisions, data retrieval, and validation results were
reviewed and verified by the team. Every figure in this README regenerates from
`train_model.py`.*
