# 🎬 OTT Viewer Retention
### Episode-Level Drop-Off Prediction & Retention Intelligence System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://ott-viewer-retention-yvjqlwsa3gmxjadipnuk6j.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/arpita-rathwa/OTT-Viewer-Retention)

> A full-stack retention analytics engine — from SQLite database and SQL analysis to multi-model ML pipeline, SHAP explainability, LSTM sequence modelling, and an intervention recommendation system — built to answer the real business question: *how do streaming platforms intervene before a viewer leaves?*

**🚀 [Try it live →](https://ott-viewer-retention-yvjqlwsa3gmxjadipnuk6j.streamlit.app/)**

---

## Real-World Business Framing

Streaming platforms invest heavily in content but viewer drop-off remains one of the most costly and least understood phenomena in OTT. This project reframes the problem from a passive prediction task into an active **Retention Intelligence System** with three real platform levers:

- **Recommendation Engine Intervention** — if episode 4 has 70% drop-off probability, promote an alternative show after episode 3
- **Content Feedback to Creators** — flag episodes with high cognitive load and weak hook strength to production teams
- **Smart Scheduling** — remove high-risk episodes from late-night recommendation slots

The output is not just a drop-off predictor — it is a system with three layers: risk scoring, intervention recommendations, and executive dashboards.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas + NumPy | Data manipulation and feature engineering |
| SQLite + sqlite3 | Relational database layer (7 tables) |
| DB Browser for SQLite | SQL query interface and analysis |
| Scikit-learn | Random Forest, Logistic Regression, preprocessing |
| XGBoost | Gradient boosting classifier and regressor |
| PyTorch | LSTM sequence model |
| SHAP | Model explainability (TreeExplainer) |
| Matplotlib | Visualization |
| Streamlit | Netflix-inspired analytics web app |
| OMDB API | Show poster fetching |
| Power BI Desktop | Executive dashboard (3 pages, `.pbix` included) |

---

## Dataset

**OTT Viewer Drop-Off & Retention Risk Dataset (v1.0)**

- **489** unique TV shows from TMDB's dynamic "Popular TV" ranking (US region)
- **33,171** episodes — one row per episode
- Viewer engagement signals synthetically generated, grounded in realistic OTT viewing behavior
- Intentionally imbalanced — reflecting real OTT ecosystems where most content falls in medium engagement

### Key Features

| Feature | Type | Description |
|---|---|---|
| `pacing_score` | Continuous (1–10) | Narrative pacing of the episode |
| `hook_strength` | Continuous (1–10) | Strength of the opening hook |
| `dialogue_density` | Ordinal (low/medium/high) | Dialogue intensity — encoded as (-1, 0, 1) |
| `visual_intensity` | Continuous (1–10) | Visual stimulation level |
| `avg_watch_percentage` | Continuous (%) | Average watch completion |
| `pause_count` | Integer | Estimated pause events |
| `rewind_count` | Integer | Estimated rewind events |
| `skip_intro` | Binary (0/1) | Whether intro is skipped |
| `cognitive_load` | Continuous (1–10) | Mental effort required |
| `attention_required` | Categorical | Low / Medium / High |
| `night_watch_safe` | Binary (0/1) | Suitable for late-night viewing |
| `drop_off` | Binary (0/1) | **Target — drop-off indicator** |
| `drop_off_probability` | Continuous (0–1) | **Target — drop-off likelihood** |
| `retention_risk` | Categorical | **Target — Low / Medium / High** |

---

## Architecture

```
OTT CSV (33,171 episodes)
        ↓
SQLite Database (7 tables)
        ↓
SQL Analysis Layer (DB Browser — 5 queries)
        ↓
ML Pipeline (3 models × 3 tasks)
        ↓
SHAP Explainability
        ↓
LSTM Sequence Model (PyTorch)
        ↓
Intervention Engine
        ↓
┌──────────────────┬─────────────────┐
│  Streamlit App   │  Power BI       │
│  (Live deployed) │  Dashboard      │
└──────────────────┴─────────────────┘
```

---

## Phase 1 — SQLite Database

Normalized schema across 3 core tables + 4 derived tables:

| Table | Rows | Description |
|---|---|---|
| `shows` | 489 | Show-level dimension table |
| `episodes` | 33,171 | Episode-level fact table |
| `retention_predictions` | 33,171 | Original dataset labels |
| `show_health` | 489 | Show health rankings (NTILE quartile-based) |
| `ml_predictions` | 33,171 | Model output predictions |
| `interventions` | 33,171 | Rule-based intervention per episode |
| `master_view` | 33,171 | Denormalized view (30 columns) for Power BI |
| `shap_importance` | 16 | SHAP feature importance scores |

---

## Phase 2 — SQL Analysis

5 business intelligence queries in DB Browser for SQLite:

- **Platform Intelligence** — avg drop risk, retention, and flagged episodes by platform
- **Retention Decay** — episode-level retention curve (capped at episode 20)
- **Genre Risk Analysis** — which genres retain viewers best
- **Intervention Queue** — all high-risk episodes with recommended actions
- **Show Health Rankings** — 489 shows classified by cancellation risk using NTILE quartiles

### Show Health Distribution
```
Healthy     223 shows   ████████████████████████
Watch       135 shows   ██████████████
Critical    113 shows   ████████████
At Risk      18 shows   ██
```

---

## Phase 3 — ML Pipeline

### Feature Engineering
- Ordinal encoding for `dialogue_density` → `{low: -1, medium: 0, high: 1}` preserving order
- Label encoding for `attention_required`, `platform`, `genre`
- StandardScaler applied for Logistic Regression only

### Model Comparison

**Task 1 — Drop-off Classification (Binary)**

| Model | Accuracy |
|---|---|
| Random Forest | 99.52% |
| XGBoost | 99.70% |
| **Logistic Regression** | **99.83%** ✅ Best |

**Task 2 — Retention Risk Classification (Multi-class)**

| Model | Accuracy |
|---|---|
| Random Forest | 99.43% |
| **XGBoost** | **99.73%** ✅ Best |
| Logistic Regression | 99.68% |

**Task 3 — Drop-off Probability Regression**

| Model | MAE |
|---|---|
| **Random Forest** | **0.0009** ✅ Best |
| XGBoost | 0.0016 |

> High accuracy is expected due to the synthetic nature of the dataset. On real platform data, models would show lower but more meaningful accuracy with genuine signal complexity.

---

## Phase 4 — SHAP Explainability

SHAP (SHapley Additive exPlanations) applied on Random Forest using TreeExplainer on 5,000 episode sample.

### Feature Importance Ranking

| Rank | Feature | Importance |
|---|---|---|
| 1 | **Avg Watch %** | 0.1203 |
| 2 | **Cognitive Load** | 0.0435 |
| 3 | **Pacing Score** | 0.0310 |
| 4 | **Hook Strength** | 0.0302 |
| 5 | **Pause Count** | 0.0270 |
| 6 | Attention Required | 0.0230 |
| 7 | Dialogue Density | 0.0179 |
| 8 | Skip Intro | 0.0153 |
| ... | ... | ... |
| 15 | Season No. | 0.0000 |
| 16 | Night Watch Safe | 0.0000 |

### Key Insights
- **Avg Watch %** is the dominant predictor — current episode completion directly signals future drop-off
- **Cognitive Load** is #2 — mentally demanding episodes drive churn significantly
- **Platform, Genre, and Season** have near-zero importance — drop-off is driven by episode content, not metadata

---

## Phase 5 — LSTM Sequence Model (PyTorch)

Unlike classical models that treat each episode independently, the LSTM treats episodes as a **temporal sequence per show** — capturing how viewing behavior evolves across a season.

### Architecture
```
Input: 5-episode context window × 12 features
    ↓
LSTM (hidden=64, layers=2, dropout=0.3)
    ↓
FC: Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid
    ↓
Output: Drop-off probability
```

**Total parameters: 55,361**

### Results vs Classical Models

| Model | Accuracy | Drop-off Recall |
|---|---|---|
| Logistic Regression | 99.83% | Low (majority class bias) |
| XGBoost | 99.73% | Low (majority class bias) |
| **LSTM** | **48%** | **63%** |

### Why LSTM Underperforms Here — and Why That Matters

Classical models achieve 99%+ because the synthetic dataset has clean linear patterns with no genuine temporal complexity. The LSTM's lower accuracy but **higher recall (63%)** on the minority drop-off class reveals that it is actually learning sequential patterns — it catches more real drop-offs at the cost of more false alarms.

> On real platform data with genuine sequential viewing dependencies, LSTM would be the superior architecture. This comparison is itself a strong portfolio insight — demonstrating understanding of when deep learning adds value vs. when classical models suffice.

### Training Details
- Class imbalance handled via **WeightedRandomSampler**
- Gradient clipping (`max_norm=1.0`) to prevent exploding gradients
- NaN handling via `nan_to_num` on sequence features
- 15 epochs, Adam optimizer, StepLR scheduler

---

## Phase 6 — Intervention Engine

Rule-based engine mapping risk scores to actionable platform interventions:

| Intervention | Episodes | Trigger |
|---|---|---|
| No intervention needed | 20,129 | Low risk |
| Remove from late-night slot | 9,735 | `night_watch_safe=0` AND `prob > 0.50` |
| Flag to content team | 2,257 | `cognitive_load > 7` AND `prob > 0.55` |
| Promote alternative show | 810 | `hook_strength < 4` AND `prob > 0.65` |
| Insert skip-recap option | 240 | `pacing_score < 4` AND `prob > 0.65` |

---

## Phase 7 — Streamlit App

Netflix-inspired dark UI built with Hanken Grotesk typography and a cinematic data aesthetic.

**3 Tabs:**
- **Episode Scorer** — select any show and episode → get real-time risk score, intervention recommendation, episode context cards, and OMDB poster
- **Platform Intel** — ranked platform table with volatility bars and risk classification
- **Intervention Queue** — priority queue of flagged episodes with filterable intervention types

**[→ Open Live App](https://ott-viewer-retention-yvjqlwsa3gmxjadipnuk6j.streamlit.app/)**

---

## Phase 8 — Power BI Dashboard

3-page executive dashboard (`.pbix` file included in repo — open in Power BI Desktop):

- **Page 1 — Executive Overview** — Platform KPI cards, platform risk bar, genre risk, top 10 riskiest shows
- **Page 2 — Retention Intelligence** — Retention decay line chart, drop-offs per episode bar, intervention donut
- **Page 3 — Show Health & Actions** — Show health scatter plot, cancellation status bar, filterable show health table

---

## ML Axes

| Axis | Detail |
|---|---|
| Learning Paradigm & Training Regime | Supervised (classification + regression) + Deep Learning (LSTM) |
| Scope & Complexity | Multi-target · Classical to deep · Research-grade comparison |
| Data Modality | Tabular (episode-level behavioral + content signals) |
| Explainability & Impact | High — SHAP-driven, intervention engine, business-facing, deployed |

---

## Scope for Fine-Tuning & Future Improvements

**Model Level**
- **LightGBM** — faster gradient boosting alternative, often outperforms XGBoost on tabular data
- **Hyperparameter tuning** — Optuna or GridSearchCV across all models
- **Transformer-based sequence model** — replace LSTM with attention mechanism for better long-range dependencies
- **Ensemble stacking** — combine Logistic Regression + XGBoost + LSTM predictions

**Feature Level**
- **Multi-season data** — current dataset is season 1 only; season fatigue analysis pending real multi-season data
- **Rolling retention rate** — cumulative drop-off trajectory per show as a feature
- **Interaction terms** — `cognitive_load × hook_strength`, `pacing_score × episode_duration`

**System Level**
- **Real user data** — replace synthetic behavioral signals with actual platform engagement data
- **A/B testing framework** — measure whether interventions actually reduce churn
- **Real-time scoring API** — FastAPI endpoint for live episode risk scoring
- **Feedback loop** — model retraining on intervention outcomes

**Dashboard Level**
- Power BI Season Fatigue analysis (pending multi-season data)
- Episode risk heatmap (season × episode matrix)
- SHAP importance embedded in Power BI via Python visual
