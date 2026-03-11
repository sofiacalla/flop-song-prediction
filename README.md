# Spotify Hit Song Prediction — A&R Marketing Optimizer

**Team:** Akase E., Miguel O., Nicolla E., Sofia C., Uma K., Yesenia P.  
**Course:** TMP 422 | Data Analytics & Decision Making | Spring 2026

---

## Problem

Record labels spend ~$50K in marketing per debut single with no reliable way to predict which tracks will flop. On a typical 1,000-release annual slate, roughly 43.5% are Tier 0 flops — translating to **$21.7M/year in blind marketing waste**. A&R teams still rely on artist popularity thresholds (e.g., "promote if artist popularity > 70") which overlap heavily across tiers and miss actual hits below that cutoff.

## Solution

A Random Forest classifier that screens new releases into **Flop / Moderate / Hit** tiers *before* any marketing budget is committed. The model acts as a first-pass filter — flagging ~33% of songs as likely flops with **71% precision** — so A&R teams only review the ~667 tracks that cleared a data-driven gate. Hit and Moderate songs are still routed to human review; the model replaces intuition at the bottom, not at the top.

**Projected net business value: ~$14.4M/year** after accounting for leakage costs and operational expenses.

---

## Pipeline Overview (`HitSongPipeline_Unified.py`)

The unified script consolidates three homework notebooks (GHW3 → GHW4 → GHW5) into a single sequential pipeline. Each part builds on the dataframe produced by the previous one.

### Part 1 — Data Cleaning & Feature Engineering

| Task | What it does |
|------|-------------|
| **T1** | Loads `songs.csv` (550K rows) and `artists.csv` (71K rows), audits blanks, replaces missing `name`/`album_name` with `"Unknown"` |
| **T2** | Removes duplicates — songs by `(name, artists, album_name)`, artists by `name` — dropping all members of each duplicate group |
| **T3** | Engineers `lyric_word_count` and `niche_genre_count` from text fields, then drops raw text columns (`lyrics`, `artist_ids`, `niche_genres`, `total_artist_followers`) to prevent leakage |
| **T4** | One-hot encodes `genre` (10 categories) into binary indicator columns |
| **T5** | Computes `tempo_deviation_bpm` — each track's tempo minus its genre's median |
| **T6** | Creates `mood_energy_interaction` = `valence × energy` |
| **T7** | Bins raw `popularity` (0–100) into an ordinal target with 4 tiers: 0 = Unpopular (≤10), 1 = Moderate (11–30), 2 = Strong (31–49), 3 = Hit (≥50, top 5%) |
| **T8** | Creates `party_score` = `energy × danceability` |
| **T9** | Adds `name_char_count` and `album_char_count`, runs OLS to test significance, then drops all text identifiers and renames targets |

**Output:** `dfsongs_final.csv` — a modeling-ready dataset with all numeric features, one-hot genres, engineered interactions, and an ordinal target column.

### Part 2 — Modeling & Precision-Recall Analysis

1. **Baseline 4-Class Random Forest** — Trains a `StandardScaler → RandomForestClassifier(n=200)` pipeline on the 4-tier target. Includes automated leakage checks (identical columns, high-correlation scan, single-feature dominance test). Produces a classification report and confusion matrix.

2. **Decision Tree Visualization** — Fits a full-depth `DecisionTreeClassifier` and exports a Graphviz SVG (capped at depth 6) to inspect which features and split points drive tier predictions.

3. **Precision–Recall Curve (Tier 3)** — Binary PR curve for Hit detection to visualize the precision/recall trade-off the team faces.

4. **3-Class Collapsed Model** — The key modeling decision: merges Tiers 1+2 into "Moderate" to improve class balance (Flop / Moderate / Hit). Trains with `class_weight="balanced"` and applies **probability thresholding** (Flop ≥ 0.75, Hit ≥ 0.50, Moderate ≥ 0.50) to make predictions more conservative. Reports baseline vs. thresholded classification metrics, feature importances, and hit-class misclassification breakdown.

5. **PR Comparison Plots** — Per-class precision-recall curves showing baseline vs. optimized operating points for all three classes, plus a dedicated Hit-class PR plot.

### Part 3 — Failure Mode Analysis

- **Prediction summary** — Counts and percentages by predicted vs. actual class
- **Hit False Positives** — Songs wrongly predicted as hits; feature-level comparison against true positives with cost estimate ($50K × count)
- **Flop False Positives** — Songs wrongly dismissed as flops, flagging any real hits that were killed
- **Hit False Negatives** — Missed hits; where they were predicted instead and how their features differ from caught hits
- **Cost summary** — Total wasted marketing spend from false positives
- **Bias check** — Accuracy and hit-precision segmented by energy, danceability, and artist popularity bands
- **4-panel visualization** — Top-feature distributions, FP/FN bar chart, feature importance, and thresholded confusion matrix heatmap

---

## Key Results

| Metric | Baseline (4-class) | Optimized (3-class + thresholds) |
|--------|-------------------|--------------------------------|
| Overall accuracy | 54.4% | **65.8%** |
| Flop precision | 59.7% | **70.9%** |
| Hit precision | 43.9% | **45.0%** |
| Hit recall | 14.5% | 8.7% (deliberate — reduces false positives) |

**Top predictive features:** `avg_artist_popularity`, `lyric_word_count`, `year`, `liveness`, `loudness`, `danceability`, `acousticness`

**"Aha" insight:** Loudness and acousticness — not artist name or label backing — are the strongest predictors of whether a track will flop. A high-profile artist releasing the wrong-sounding track is still a marketing risk, and the model catches that.

---

## Deployment Strategy

The model is designed for a **two-stage A&R workflow:**

1. **Stage 1 — Automated Filter:** Every new release runs through the model. Songs flagged as Flop (≥75% probability) are auto-deprioritized before any marketing dollars are committed.
2. **Stage 2 — Human Review:** Moderate and Hit predictions are surfaced on an internal A&R dashboard with tier labels, confidence scores, and key audio signals. A&R approves budget before spend.

**Guardrails:** input validation (13 Spotify API features, genre coverage, new-artist flag, range checks), weekly monitoring (flop detection rate, flop precision, rejected-hit leakage rate), and semi-annual retraining.

---

## Repository Structure

```
├── HitSongPipeline_Unified.py   # Full pipeline: cleaning → modeling → failure analysis
├── README.md
├── data/
│   ├── songs.csv                 # Raw Spotify song data (550K rows)
│   └── artists.csv               # Raw artist metadata (71K rows)
└── outputs/
    ├── dfsongs_final.csv          # Cleaned, feature-engineered dataset
    ├── precision_recall_comparison.png
    ├── hit_precision_recall.png
    └── failure_mode_analysis.png
```

---

## How to Run

```bash
# Requires Python 3.8+
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels graphviz

# Place songs.csv and artists.csv in /content/ (or update paths in the script)
python HitSongPipeline_Unified.py
```

> **Note:** The script uses `display()` and `IPython.display` for Graphviz rendering. Run in Jupyter/Colab for full output, or comment out those lines for terminal execution.

---

## AI Disclosure

ChatGPT and Claude were used to assist with research, grammar correction, and code debugging throughout the project.

## Data
- `artists.csv` — included in repo
- `songs.csv` — too large for GitHub, [download here](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres/data)
