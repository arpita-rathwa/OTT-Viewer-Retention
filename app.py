import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import torch
import torch.nn as nn
import pickle
import requests
import streamlit.components.v1 as components

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="RetentionIQ — OTT Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Poster Fetching ──────────────────────────────────────
OMDB_API_KEY = "b27d82ce"

@st.cache_data
def fetch_poster(title):
    try:
        response = requests.get(
            f'http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}',
            timeout=5
        )
        data = response.json()
        poster = data.get('Poster', '')
        if not poster or poster == 'N/A':
            return "https://via.placeholder.com/300x450/201f1f/5e3f3b?text=No+Poster"
        return poster
    except:
        return "https://via.placeholder.com/300x450/201f1f/5e3f3b?text=No+Poster"

# ── LSTM Model ───────────────────────────────────────────
class DropOffLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(DropOffLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# ── Load Assets ──────────────────────────────────────────
@st.cache_resource
def load_models():
    lstm = DropOffLSTM(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
    lstm.load_state_dict(torch.load('lstm_model.pth', map_location='cpu', weights_only=True))
    lstm.eval()
    with open('lstm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return lstm, scaler

@st.cache_data
def load_data():
    conn = sqlite3.connect('ott_database.db')
    shows = pd.read_sql_query("SELECT DISTINCT title, show_id FROM shows ORDER BY title", conn)
    master = pd.read_sql_query("SELECT * FROM master_view", conn)
    shap_df = pd.read_sql_query("SELECT * FROM shap_importance", conn)
    conn.close()
    return shows, master, shap_df

lstm_model, scaler = load_models()
shows_df, master_df, shap_df = load_data()

total_shows = master_df['show_id'].nunique()
total_eps = len(master_df)
flagged = (master_df['intervention'] != 'No intervention needed').sum()
avg_risk = master_df['predicted_probability'].mean()

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background-color: #131313;
    color: #e5e2e1;
    font-family: 'Hanken Grotesk', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1rem 2rem 1rem !important; max-width: 100% !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #131313; }
::-webkit-scrollbar-thumb { background: #353534; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #e50914; }

/* Nav */
.topnav {
    background: #131313;
    border-bottom: 1px solid #5e3f3b;
    padding: 12px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-logo {
    font-size: 28px;
    font-weight: 800;
    color: #e50914;
    letter-spacing: -0.02em;
    cursor: pointer;
}

.nav-links { display: flex; gap: 32px; }

.nav-link {
    font-size: 16px;
    color: #e9bcb6;
    text-decoration: none;
    transition: color 0.2s;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    padding-bottom: 4px;
}

.nav-link.active {
    color: #e5e2e1;
    border-bottom: 2px solid #e50914;
}

/* Main */
.main-content { padding: 0 48px 64px 48px; }

/* Hero */
.hero { padding: 40px 0 32px 0; }

.hero-eyebrow {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #e9bcb6;
    opacity: 0.6;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #e5e2e1;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 32px;
}

/* Stat Cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px;
    margin-bottom: 40px;
}

.stat-card {
    background: #201f1f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color 0.2s;
}

.stat-card:hover { border-color: #e50914; }

.stat-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e9bcb6;
}

.stat-value {
    font-size: 32px;
    font-weight: 700;
    color: #e5e2e1;
    line-height: 1.2;
}

.stat-value.red { color: #e50914; }

.stat-bar {
    width: 100%;
    height: 4px;
    background: #353534;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}

.stat-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: #e50914;
}

/* Section Header */
.section-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e9bcb6;
    opacity: 0.8;
    margin-bottom: 8px;
}

.section-headline {
    font-size: 32px;
    font-weight: 700;
    color: white;
    letter-spacing: -0.01em;
    margin-bottom: 24px;
}

/* Glass Card */
.glass-card {
    background: #201f1f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 24px;
}

/* Split Layout */
.split-layout {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 24px;
    align-items: start;
}

/* Select Panel */
.select-panel {
    background: #201f1f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 24px;
    font-weight: 700;
    color: #e5e2e1;
}

.field-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e9bcb6;
    margin-bottom: 8px;
}

/* Poster */
.poster-container {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}

.poster-container img { width: 100%; display: block; }

/* Signal Bars */
.signal-row {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.signal-item { display: flex; flex-direction: column; gap: 6px; }

.signal-meta {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}

.signal-name {
    color: #e9bcb6;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.signal-val { color: #e5e2e1; font-weight: 500; }

.signal-bar {
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
}

.signal-bar-fill { height: 100%; border-radius: 2px; }

/* Risk Panel */
.risk-panel {
    background: #201f1f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 32px;
}

.risk-big {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
}

.risk-pct {
    font-size: 72px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}

.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 12px;
}

.badge-high { background: rgba(229,9,20,0.15); color: #ff4444; border: 1px solid rgba(229,9,20,0.4); }
.badge-medium { background: rgba(255,165,0,0.12); color: #ffaa00; border: 1px solid rgba(255,165,0,0.3); }
.badge-low { background: rgba(0,200,100,0.12); color: #00cc66; border: 1px solid rgba(0,200,100,0.3); }

.risk-meta { text-align: right; }

.risk-meta-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e9bcb6;
    opacity: 0.6;
    margin-bottom: 6px;
}

.risk-meta-value {
    font-size: 24px;
    font-weight: 700;
    color: white;
}

.risk-meta-sub {
    font-size: 14px;
    color: #e9bcb6;
    margin-top: 4px;
}

/* Intervention Box */
.intervention-box {
    background: linear-gradient(135deg, rgba(229,9,20,0.08), rgba(229,9,20,0.03));
    border: 1px solid rgba(229,9,20,0.25);
    border-left: 3px solid #e50914;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
}

.intervention-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e50914;
    margin-bottom: 6px;
}

.intervention-text {
    font-size: 16px;
    color: rgba(255,255,255,0.85);
}

/* Episode Cards */
.ep-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-top: 8px;
}

.ep-card {
    background: #1c1b1b;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 16px;
    position: relative;
    overflow: hidden;
}

.ep-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--c);
}

.ep-num {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 8px;
}

.ep-pct {
    font-size: 28px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}

.ep-sub {
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    margin-top: 6px;
}

.ep-progress {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 10px;
}

.ep-progress-fill { height: 100%; border-radius: 2px; }

/* Platform Table */
.table-container {
    background: #1c1b1b;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
}

.table-header {
    display: grid;
    grid-template-columns: 48px 1fr 120px 120px 200px;
    gap: 24px;
    padding: 12px 24px;
    border-bottom: 1px solid rgba(94,63,59,0.5);
    background: #2a2a2a;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #e9bcb6;
}

.table-row {
    display: grid;
    grid-template-columns: 48px 1fr 120px 120px 200px;
    gap: 24px;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    align-items: center;
    transition: background 0.15s;
    cursor: pointer;
}

.table-row:hover { background: #2a2a2a; }

.table-rank {
    font-size: 20px;
    font-weight: 800;
    color: rgba(255,255,255,0.2);
}

.table-platform { font-size: 16px; font-weight: 700; color: white; }

.table-retention {
    font-size: 16px;
    font-weight: 700;
    color: white;
    text-align: center;
}

.risk-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    text-align: center;
}

.pill-critical { background: #e50914; color: white; }
.pill-high { background: rgba(160,2,22,0.8); color: white; }
.pill-medium { background: rgba(255,165,0,0.2); color: #ffaa00; border: 1px solid rgba(255,165,0,0.3); }
.pill-low { background: rgba(0,200,100,0.15); color: #00cc66; border: 1px solid rgba(0,200,100,0.3); }

.vol-bar {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vol-track {
    flex: 1;
    height: 8px;
    background: #353534;
    border-radius: 999px;
    overflow: hidden;
}

.vol-fill { height: 100%; background: #e50914; border-radius: 999px; }
.vol-val { font-size: 14px; font-weight: 500; color: #e9bcb6; min-width: 36px; }

/* Queue */
.queue-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 24px;
    background: #1c1b1b;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    margin-bottom: 8px;
    transition: background 0.15s;
}

.queue-item:hover { background: #2a2a2a; }

.queue-rank {
    font-size: 20px;
    font-weight: 800;
    color: rgba(255,255,255,0.2);
    min-width: 36px;
}

.queue-accent {
    width: 3px;
    height: 40px;
    border-radius: 2px;
    flex-shrink: 0;
}

.queue-info { flex: 1; }
.queue-title { font-size: 14px; font-weight: 700; color: white; }
.queue-meta { font-size: 12px; color: #e9bcb6; margin-top: 2px; opacity: 0.6; }
.queue-action { font-size: 12px; color: #e9bcb6; max-width: 220px; text-align: right; opacity: 0.7; }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(to right, rgba(229,9,20,0.3), transparent);
    margin: 24px 0;
}

/* Streamlit overrides */
.stButton > button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}

.stButton > button:hover { background: #b81d24 !important; }

.stSelectbox > div > div {
    background: #2a2a2a !important;
    border: 1px solid #5e3f3b !important;
    border-radius: 8px !important;
    color: #e5e2e1 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
}

.stSelectbox > div > div:focus-within { border-color: #e50914 !important; }

.stMultiSelect > div > div {
    background: #2a2a2a !important;
    border: 1px solid #5e3f3b !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #5e3f3b;
    gap: 0;
    padding: 0 48px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #e9bcb6;
    font-family: 'Hanken Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 500;
    padding: 16px 24px;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    border-bottom: 2px solid #e50914 !important;
    background: transparent !important;
}

label, .stSelectbox label, .stMultiSelect label {
    color: #e9bcb6 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="background:#131313;border-bottom:1px solid #5e3f3b;
            padding:12px 48px;display:flex;align-items:center;gap:48px;">
    <span style="font-family:'Hanken Grotesk',sans-serif;font-size:28px;
                 font-weight:800;color:#e50914;letter-spacing:-0.02em;">RETENTIONIQ</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  EPISODE SCORER", "📡  PLATFORM INTEL", "⚡  INTERVENTION QUEUE"])

# ══════════════════════════════════════════════════════════
# TAB 1 — Episode Scorer
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">Internal Dashboard</div>
        <div class="hero-title">Real-time Cinematic Retention Metrics</div>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Shows Tracked</div>
                <div class="stat-value red">{total_shows}</div>
                <div class="stat-bar"><div class="stat-bar-fill" style="width:75%;"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Episodes Analysed</div>
                <div class="stat-value">{total_eps:,}</div>
                <div class="stat-bar"><div class="stat-bar-fill" style="width:50%;background:white;"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Episodes Flagged</div>
                <div class="stat-value red">{flagged:,}</div>
                <div class="stat-bar"><div class="stat-bar-fill" style="width:100%;"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Drop Risk</div>
                <div class="stat-value">{avg_risk:.0%}</div>
                <div class="stat-bar"><div class="stat-bar-fill" style="width:{avg_risk*100}%;"></div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("""
        <div class="section-title">Select Content</div>
        """, unsafe_allow_html=True)

        selected_show = st.selectbox("SHOW", shows_df['title'].values)
        show_id = shows_df[shows_df['title'] == selected_show]['show_id'].values[0]
        show_eps = master_df[master_df['show_id'] == show_id].sort_values('episode_number')

        selected_ep = st.selectbox(
            "EPISODE",
            show_eps['episode_number'].values,
            format_func=lambda x: f"Episode {x:02d}"
        )

        analyse = st.button("▶  ANALYSE EPISODE")

        ep_data = show_eps[show_eps['episode_number'] == selected_ep].iloc[0]

        # Poster
        with st.spinner(''):
            poster_url = fetch_poster(selected_show)

        st.markdown(f"""
        <div class="poster-container" style="margin:16px 0;">
            <img src="{poster_url}"
                 onerror="this.src='https://via.placeholder.com/300x450/201f1f/5e3f3b?text=No+Poster'"
            />
        </div>
        <div class="divider"></div>
        <div class="section-title" style="margin-bottom:16px;">Episode Signals</div>
        """, unsafe_allow_html=True)

        signals = [
            ("Hook Strength", ep_data['hook_strength'], 10, "#e50914"),
            ("Pacing Score", ep_data['pacing_score'], 10, "#ff6b35"),
            ("Cognitive Load", ep_data['cognitive_load'], 10, "#ffd700"),
            ("Watch Completion", ep_data['avg_watch_percentage'], 100, "#00cc66"),
        ]

        signal_html = '<div class="signal-row">'
        for label, val, max_val, color in signals:
            pct = (val / max_val) * 100
            signal_html += f"""
            <div class="signal-item">
                <div class="signal-meta">
                    <span class="signal-name">{label}</span>
                    <span class="signal-val">{val}</span>
                </div>
                <div class="signal-bar">
                    <div class="signal-bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>"""
        signal_html += '</div>'
        st.markdown(signal_html, unsafe_allow_html=True)

    with col_right:
        if analyse:
            risk_prob = ep_data['predicted_probability']
            risk_tier = ep_data['predicted_risk']
            intervention = ep_data['intervention']
            cancel_status = ep_data['cancellation_status']

            if risk_prob >= 0.65:
                risk_color = "#e50914"
                badge_class = "badge-high"
                risk_icon = "⬤"
            elif risk_prob >= 0.50:
                risk_color = "#ffaa00"
                badge_class = "badge-medium"
                risk_icon = "⬤"
            else:
                risk_color = "#00cc66"
                badge_class = "badge-low"
                risk_icon = "⬤"

            st.markdown(f"""
            <div class="section-title" style="margin-bottom:8px;">Risk Assessment</div>
            <div class="risk-panel">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{risk_color};border-radius:12px 12px 0 0;opacity:0.8;"></div>
                <div class="risk-big">
                    <div>
                        <div style="font-size:12px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#e9bcb6;opacity:0.6;margin-bottom:8px;">Drop-off Probability</div>
                        <div class="risk-pct" style="color:{risk_color};">{risk_prob:.0%}</div>
                        <div class="risk-badge {badge_class}">{risk_icon} {risk_tier.upper()} RISK</div>
                    </div>
                    <div class="risk-meta">
                        <div class="risk-meta-label">Show Status</div>
                        <div class="risk-meta-value">{cancel_status}</div>
                        <div class="risk-meta-sub">S{int(ep_data['season_number']):02d} · E{int(selected_ep):02d}</div>
                    </div>
                </div>
                <div class="intervention-box">
                    <div class="intervention-label">⚡ Recommended Action</div>
                    <div class="intervention-text">{intervention}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="section-title" style="margin:24px 0 12px 0;">Episode Context — Last 5 Episodes</div>
            """, unsafe_allow_html=True)

            context = show_eps[show_eps['episode_number'] <= selected_ep].tail(5)
            ep_html = '<div class="ep-grid">'
            for _, row in context.iterrows():
                prob = row['predicted_probability']
                c = "#e50914" if prob >= 0.65 else "#ffaa00" if prob >= 0.50 else "#00cc66"
                ep_html += f"""
                <div class="ep-card" style="--c:{c};">
                    <div class="ep-num">EP {int(row['episode_number']):02d}</div>
                    <div class="ep-pct" style="color:{c};">{prob:.0%}</div>
                    <div style="font-size:11px;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.08em;">Drop Risk</div>
                    <div class="ep-sub">{row['avg_watch_percentage']:.0f}% watched</div>
                    <div class="ep-progress">
                        <div class="ep-progress-fill" style="width:{prob*100}%;background:{c};"></div>
                    </div>
                </div>"""
            ep_html += '</div>'
            st.markdown(ep_html, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:500px;flex-direction:column;gap:16px;opacity:0.3;">
                <span class="material-symbols-outlined" style="font-size:64px;color:#e50914;">movie</span>
                <div style="font-size:32px;font-weight:800;letter-spacing:-0.01em;color:white;">
                    SELECT AN EPISODE
                </div>
                <div style="font-size:16px;color:#e9bcb6;">
                    Risk scores and interventions will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — Platform Intel
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Platform Intelligence</div>
        <div class="hero-title">PLATFORM RISK RANKING</div>
    </div>
    """, unsafe_allow_html=True)

    platform_stats = master_df.groupby('platform').agg(
        total_shows=('show_id', 'nunique'),
        avg_retention=('avg_watch_percentage', 'mean'),
        avg_drop_risk=('predicted_probability', 'mean'),
        episodes_flagged=('intervention', lambda x: (x != 'No intervention needed').sum())
    ).round(3).reset_index().sort_values('avg_drop_risk', ascending=False).reset_index(drop=True)

    # KPI Cards
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Total Shows</div>
            <div class="stat-value">{total_shows}</div>
            <div class="stat-bar"><div class="stat-bar-fill" style="width:75%;"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Episodes</div>
            <div class="stat-value">{total_eps:,}</div>
            <div class="stat-bar"><div class="stat-bar-fill" style="width:50%;background:white;"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Retention</div>
            <div class="stat-value">{master_df['avg_watch_percentage'].mean():.1f}%</div>
            <div class="stat-bar"><div class="stat-bar-fill" style="width:{master_df['avg_watch_percentage'].mean()}%;background:#00cc66;"></div></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Episodes Flagged</div>
            <div class="stat-value red">{flagged:,}</div>
            <div class="stat-bar"><div class="stat-bar-fill" style="width:100%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Table
    max_risk = platform_stats['avg_drop_risk'].max()
    min_risk = platform_stats['avg_drop_risk'].min()

    table_html = """
    <div class="table-container">
        <div class="table-header">
            <div>RANK</div>
            <div>PLATFORM NAME</div>
            <div style="text-align:center;">AVG RETENTION</div>
            <div style="text-align:center;">RISK SCORE</div>
            <div>VOLATILITY TREND</div>
        </div>
        <div>
    """

    for i, row in platform_stats.iterrows():
        norm = (row['avg_drop_risk'] - min_risk) / (max_risk - min_risk + 0.001)
        bar_pct = norm * 100

        if row['avg_drop_risk'] >= 0.51:
            pill = '<span class="risk-pill pill-critical">CRITICAL</span>'
        elif row['avg_drop_risk'] >= 0.49:
            pill = '<span class="risk-pill pill-high">HIGH</span>'
        elif row['avg_drop_risk'] >= 0.47:
            pill = '<span class="risk-pill pill-medium">MEDIUM</span>'
        else:
            pill = '<span class="risk-pill pill-low">LOW</span>'

        table_html += f"""
        <div class="table-row">
            <div class="table-rank">{i+1:02d}</div>
            <div class="table-platform">{row['platform']}</div>
            <div class="table-retention">{row['avg_retention']:.1f}%</div>
            <div style="display:flex;justify-content:center;">{pill}</div>
            <div class="vol-bar">
                <div class="vol-track">
                    <div class="vol-fill" style="width:{bar_pct:.0f}%;"></div>
                </div>
                <span class="vol-val">{row['avg_drop_risk']:.3f}</span>
            </div>
        </div>"""

    table_html += "</div></div>"
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;700;800&display=swap" rel="stylesheet"/>
<style>
body {{ background: transparent; margin: 0; font-family: 'Hanken Grotesk', sans-serif; }}
.table-container {{ background: #1c1b1b; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; }}
.table-header {{ display: grid; grid-template-columns: 48px 1fr 120px 120px 200px; gap: 24px; padding: 12px 24px; border-bottom: 1px solid rgba(94,63,59,0.5); background: #2a2a2a; font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #e9bcb6; }}
.table-row {{ display: grid; grid-template-columns: 48px 1fr 120px 120px 200px; gap: 24px; padding: 16px 24px; border-bottom: 1px solid rgba(255,255,255,0.04); align-items: center; color: #e5e2e1; }}
.table-row:hover {{ background: #2a2a2a; }}
.table-rank {{ font-size: 20px; font-weight: 800; color: rgba(255,255,255,0.2); }}
.table-platform {{ font-size: 16px; font-weight: 700; color: white; }}
.table-retention {{ font-size: 16px; font-weight: 700; color: white; text-align: center; }}
.risk-pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 10px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; text-align: center; }}
.pill-critical {{ background: #e50914; color: white; }}
.pill-high {{ background: rgba(160,2,22,0.8); color: white; }}
.pill-medium {{ background: rgba(255,165,0,0.2); color: #ffaa00; border: 1px solid rgba(255,165,0,0.3); }}
.pill-low {{ background: rgba(0,200,100,0.15); color: #00cc66; border: 1px solid rgba(0,200,100,0.3); }}
.vol-bar {{ display: flex; align-items: center; gap: 8px; }}
.vol-track {{ flex: 1; height: 8px; background: #353534; border-radius: 999px; overflow: hidden; }}
.vol-fill {{ height: 100%; background: #e50914; border-radius: 999px; }}
.vol-val {{ font-size: 14px; font-weight: 500; color: #e9bcb6; min-width: 36px; }}
</style>
{table_html}
""", height=len(platform_stats) * 56 + 60, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — Intervention Queue
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Action Required</div>
        <div class="hero-title">INTERVENTION PRIORITY QUEUE</div>
    </div>
    """, unsafe_allow_html=True)

    intervention_types = [i for i in master_df['intervention'].unique()
                          if i != 'No intervention needed']

    selected_interventions = st.multiselect(
        "FILTER BY ACTION TYPE",
        options=intervention_types,
        default=intervention_types
    )

    queue = master_df[master_df['intervention'].isin(selected_interventions)][
        ['title', 'platform', 'season_number', 'episode_number',
         'predicted_probability', 'predicted_risk', 'intervention']
    ].sort_values('predicted_probability', ascending=False).head(30)

    for i, (_, row) in enumerate(queue.iterrows(), 1):
        prob = row['predicted_probability']
        c = "#e50914" if prob >= 0.65 else "#ffaa00" if prob >= 0.50 else "#00cc66"
        badge = "badge-high" if prob >= 0.65 else "badge-medium" if prob >= 0.50 else "badge-low"

        st.markdown(f"""
        <div class="queue-item">
            <div class="queue-rank">#{i:02d}</div>
            <div class="queue-accent" style="background:{c};"></div>
            <div class="queue-info">
                <div class="queue-title">{row['title']}</div>
                <div class="queue-meta">{row['platform']} · S{int(row['season_number']):02d}E{int(row['episode_number']):02d}</div>
            </div>
            <div class="risk-badge {badge}" style="flex-shrink:0;">{prob:.0%}</div>
            <div class="queue-action">{row['intervention']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="margin-bottom:16px;">INTERVENTION BREAKDOWN</div>', unsafe_allow_html=True)

    int_counts = master_df[
        master_df['intervention'] != 'No intervention needed'
    ]['intervention'].value_counts()
    max_count = int_counts.max()

    for intervention, count in int_counts.items():
        bar_pct = (count / max_count) * 100
        st.markdown(f"""
        <div class="queue-item" style="margin-bottom:8px;">
            <div class="queue-info">
                <div class="queue-title" style="font-size:13px;">{intervention}</div>
            </div>
            <div class="vol-bar" style="min-width:300px;">
                <div class="vol-track">
                    <div class="vol-fill" style="width:{bar_pct:.0f}%;"></div>
                </div>
                <span class="vol-val" style="color:white;font-weight:700;">{count:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)