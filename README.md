# 🎬 OTT Viewer Retention

### Episode-Level Drop-Off Prediction & Retention Risk Modelling for Streaming Platforms

> Predicting when and why viewers stop watching — using episode-level behavioral signals to model churn risk, retention dynamics, and content engagement patterns across OTT TV series.

---

## Problem Statement

Streaming platforms invest heavily in content acquisition and production, yet viewer drop-off — the point at which a user stops watching a series — remains one of the most costly and least understood phenomena in the OTT space. Traditional catalog metadata (genres, ratings, cast) fails to explain *when* viewers disengage or *why* specific episodes trigger churn.

The core question: *can episode-level behavioral and content signals predict viewer drop-off probability and retention risk with enough granularity to be actionable for content teams, recommendation engines, and scheduling decisions?*

---

## Dataset

**OTT Viewer Drop-Off & Retention Risk Dataset (v1.0)**

- 450+ unique TV shows from TMDB's dynamic "Popular TV" ranking (US region)
- Episode-level granularity — one row per episode
- Up to 4 seasons per show, all episodes included
- Viewer engagement signals are synthetically generated, grounded in realistic OTT viewing behavior assumptions
- Intentionally imbalanced — reflecting real OTT ecosystems where most content falls in medium engagement

### Key Features

| Feature | Type | Description |
|---|---|---|
| `pacing_score` | Continuous (1–10) | Narrative pacing of the episode |
| `hook_strength` | Continuous (1–10) | Strength of the opening hook |
| `dialogue_density` | Continuous (1–10) | Dialogue intensity |
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

## Technical Scope

### Task Types
- **Binary classification** — predicting whether an episode triggers viewer drop-off (`drop_off`)
- **Regression** — predicting the probability of drop-off (`drop_off_probability`)
- **Multi-class classification** — predicting retention risk tier (Low / Medium / High)
- **Clustering** — segmenting episodes or shows by engagement profile
- **Time series analysis** — modelling retention decay across episode and season progressions

### Key Analytical Questions
- At which episode do viewers start dropping off within a season?
- How does retention change from Season 1 to later seasons — does season fatigue exist?
- Do finales improve or hurt retention?
- How do pacing, hook strength, and cognitive load interact to drive churn?
- Which episodes are risky for late-night recommendation slots?
- Which content signals are the strongest predictors of retention? (SHAP)

### Modelling Challenges
- **Class imbalance** — most episodes fall in medium engagement; minority class (high drop-off) requires careful handling via SMOTE, class weighting, or threshold tuning
- **Temporal dependencies** — viewer behavior at episode N is influenced by episodes 1 to N-1; sequential models may outperform tabular ones
- **Synthetic data limitations** — behavioral signals are simulated; model generalization to real platform data requires careful assumption validation
- **Multi-target prediction** — `drop_off`, `drop_off_probability`, and `retention_risk` are correlated targets; multi-output or hierarchical modelling may be beneficial
- **Show-level vs episode-level effects** — some shows inherently retain better; models must separate show-level quality signals from episode-level triggers

### Feature Engineering Opportunities
- Rolling retention rate per show across episodes (cumulative drop-off trajectory)
- Season position encoding (early, mid, finale)
- Interaction terms: `cognitive_load × hook_strength`, `pacing_score × episode_duration`
- Platform-level aggregates (retention benchmarks per streaming service)
- Genre-conditioned normalization (cognitive load in a thriller vs. a sitcom means different things)

### Explainability
- **SHAP** — identifying which episode-level signals drive drop-off predictions the most
- Feature importance breakdown per retention risk tier
- Episode-level risk flagging for content and scheduling teams

### Evaluation Metrics
- Classification: AUC-ROC, F1 (macro), Precision-Recall curve (given imbalance)
- Regression: MAE, RMSE on `drop_off_probability`
- Business metric: % of high-risk episodes correctly flagged before they air in a recommendation slot

---

## ML Axes

| Axis | Detail |
|---|---|
| Learning Paradigm & Training Regime | Supervised — classification + regression; unsupervised clustering |
| Scope & Complexity | Multi-target · Classical to ensemble ML |
| Data Modality | Tabular (episode-level behavioral + content signals) |
| Explainability & Impact | High — SHAP-driven, content strategy and recommendation-facing |

---

## To Be Added
- [ ] Framework & libraries used
- [ ] Model selection and comparison results
- [ ] SHAP feature importance plots
- [ ] Season-level retention decay visualizations
- [ ] Drop-off risk dashboard
