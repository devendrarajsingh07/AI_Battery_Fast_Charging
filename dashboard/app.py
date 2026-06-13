"""
BatteryAI — Fast Charging Optimizer Dashboard
Streamlit front-end for the SAC-based lithium-ion charging agent.

Run from project root:
    streamlit run dashboard/app.py
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BatteryAI — Fast Charging",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# CSS  (dark engineering-style theme)
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #060b14 !important;
    color: #c9d1d9 !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1c2128;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e6edf3 !important; }

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 28px 0 20px;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    background: linear-gradient(130deg, #58a6ff 10%, #39d353 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
}
.hero-sub {
    color: #6e7681;
    font-size: 0.95rem;
    margin-top: 6px;
    letter-spacing: 0.01em;
}

/* ── KPI cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0 24px;
}
.kpi {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 16px 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.kpi:hover {
    border-color: #30363d;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi.g::before { background: linear-gradient(90deg,#00b36b,#39d353); }
.kpi.b::before { background: linear-gradient(90deg,#1f6feb,#58a6ff); }
.kpi.a::before { background: linear-gradient(90deg,#bb8009,#ffa657); }
.kpi.p::before { background: linear-gradient(90deg,#6e40c9,#bc8cff); }

.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
    font-weight: 600;
}
.kpi-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 6px 0 4px;
}
.kpi.g .kpi-val { color: #39d353; }
.kpi.b .kpi-val { color: #58a6ff; }
.kpi.a .kpi-val { color: #ffa657; }
.kpi.p .kpi-val { color: #bc8cff; }
.kpi-unit { font-size: 0.85rem; color: #484f58; }
.kpi-sub  { font-size: 0.76rem; color: #6e7681; margin-top: 2px; }

/* ── SOC progress bar ── */
.soc-wrap  { margin: 6px 0 16px; }
.soc-track {
    background: #161b22;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    border: 1px solid #21262d;
}
.soc-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #00b36b, #39d353 60%, #7ee787);
    transition: width 0.5s cubic-bezier(.4,0,.2,1);
}
.soc-label { font-size: 0.8rem; color: #6e7681; margin-bottom: 4px; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6e7681 !important;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 10px 22px 8px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #39d353 !important;
    border-bottom: 2px solid #39d353 !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity .2s, transform .15s !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

/* ── select / slider ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] {
    padding: 4px 0 !important;
}

/* ── info / status pills ── */
.pill {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pill-ok   { background: rgba(57,211,83,.12); color: #39d353; border: 1px solid rgba(57,211,83,.3); }
.pill-warn { background: rgba(255,166,87,.12); color: #ffa657; border: 1px solid rgba(255,166,87,.3); }
.pill-bad  { background: rgba(248,81,73,.12);  color: #f85149; border: 1px solid rgba(248,81,73,.3); }

/* ── section headings ── */
.sec-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #484f58;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* ── callout box ── */
.callout {
    background: #0d1117;
    border-left: 3px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 14px 0;
    color: #8b949e;
    font-size: 0.87rem;
    line-height: 1.6;
}
.callout strong { color: #c9d1d9; }

/* ── divider ── */
hr { border-color: #21262d !important; margin: 20px 0 !important; }

/* ── physics table ── */
.phys-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.phys-table th {
    text-align: left; padding: 8px 14px;
    background: #161b22; color: #8b949e;
    font-weight: 600; font-size: 0.74rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #21262d;
}
.phys-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #1c2128;
    color: #c9d1d9;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
}
.phys-table tr:hover td { background: #0d1117; }

/* ── about card ── */
.about-card {
    background: linear-gradient(160deg, #0d1117, #161b22);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 28px 32px;
    margin: 12px 0;
}
.about-card h3 { color: #e6edf3; font-size: 1.1rem; margin-bottom: 10px; }
.about-card p  { color: #8b949e; font-size: 0.88rem; line-height: 1.65; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8b949e", size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e")),
    hoverlabel=dict(
        bgcolor="#161b22", font_color="#c9d1d9",
        font_family="Space Mono", bordercolor="#30363d"
    ),
)


def _axis(title):
    return dict(
        title=title,
        gridcolor="#1c2128",
        zerolinecolor="#21262d",
        tickfont=dict(color="#6e7681", size=11),
        title_font=dict(color="#6e7681", size=11),
    )


def _hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    """Convert a #rrggbb hex string to 'rgba(r,g,b,a)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def line_chart(x, y, title, y_title, color, threshold=None, threshold_label=""):
    """Single trace area chart with optional horizontal limit line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.10),
        hovertemplate=f"Step %{{x}}<br>{y_title}: %{{y:.3f}}<extra></extra>",
    ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(dash="dot", color="#f85149", width=1.5),
            annotation_text=threshold_label,
            annotation_font=dict(color="#f85149", size=10),
        )

    fig.update_layout(title=dict(text=title, font=dict(color="#e6edf3", size=13)),
                      xaxis=_axis("Time step"), yaxis=_axis(y_title),
                      **PLOT_LAYOUT)
    return fig


def kpi_html(label, value, unit, sub, cls):
    return f"""
    <div class="kpi {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val">{value}<span class="kpi-unit"> {unit}</span></div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def status_pill(val, good_thresh, warn_thresh, invert=False):
    """Return colored status pill HTML. invert=True means lower is worse."""
    if invert:
        ok = val >= good_thresh
        warn = good_thresh > val >= warn_thresh
    else:
        ok = val <= good_thresh
        warn = good_thresh < val <= warn_thresh
    if ok:
        return '<span class="pill pill-ok">SAFE</span>'
    elif warn:
        return '<span class="pill pill-warn">CAUTION</span>'
    return '<span class="pill pill-bad">CRITICAL</span>'


def soc_bar(soc: float):
    pct = int(soc * 100)
    color = "#39d353" if soc < 0.75 else "#ffa657" if soc < 0.90 else "#f85149"
    fill_style = f"width:{pct}%; background: {color};"
    return f"""
    <div class="soc-wrap">
        <div class="soc-label">State of Charge — {pct}%</div>
        <div class="soc-track"><div class="soc-fill" style="{fill_style}"></div></div>
    </div>"""


# ──────────────────────────────────────────────────────────────────────
# simulation
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained SAC model (cached so it doesn't reload on every interaction)."""
    try:
        from stable_baselines3 import SAC
        model_path = os.path.join(ROOT, "results", "models", "continuous_sac_model")
        model = SAC.load(model_path)
        return model, True
    except Exception as e:
        return None, False


def run_episode(c_rate: float, use_rl: bool, rl_model) -> dict:
    """
    Run one full charging episode and return timestep history.
    Falls back to manual constant-current if RL model isn't available.
    """
    from env.continuous_pybamm_env import ContinuousBatteryEnv

    env = ContinuousBatteryEnv()
    state, _ = env.reset()

    hist = {k: [] for k in ("soc", "voltage", "temperature", "soh", "current", "reward")}

    done = False
    safety_cap = 600   # prevent infinite loops during edge cases

    while not done and len(hist["soc"]) < safety_cap:
        if use_rl and rl_model is not None:
            action, _ = rl_model.predict(state, deterministic=True)
        else:
            action = np.array([c_rate], dtype=np.float32)

        state, reward, done, _, info = env.step(action)

        hist["soc"].append(info["soc"])
        hist["voltage"].append(info["voltage"])
        hist["temperature"].append(info["temperature"])
        hist["soh"].append(info["soh"])
        hist["current"].append(info["current"])
        hist["reward"].append(reward)

    return hist


# ──────────────────────────────────────────────────────────────────────
# sidebar
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ BatteryAI")
    st.markdown('<p style="color:#484f58;font-size:0.8rem;">SAC Charging Optimizer</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sec-title">Simulation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Controller",
        ["🤖  RL Agent (trained SAC)", "🔧  Manual — constant C-rate"],
        label_visibility="collapsed",
    )
    use_rl = "RL" in mode

    st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
    c_rate = st.slider(
        "Charging C-rate", min_value=0.5, max_value=3.0, value=1.7,
        step=0.1, help="Applied current as a multiple of nominal capacity (1C = 3 A)"
    )
    if use_rl:
        st.caption("C-rate shown as reference only — RL agent adapts dynamically.")

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", width='stretch')
    st.markdown("")
    st.markdown(
        '<div class="callout">'
        '<strong>How it works</strong><br>'
        'The SAC agent controls charging current at each timestep (60 s), '
        'balancing <em>speed</em>, <em>heat</em>, and <em>cell aging</em>.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-title">Model status</div>', unsafe_allow_html=True)
    rl_model, model_ok = load_model()
    if model_ok:
        st.markdown('<span class="pill pill-ok">Model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-warn">Model not found — manual only</span>', unsafe_allow_html=True)
        use_rl = False


# ──────────────────────────────────────────────────────────────────────
# header
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 class="hero-title">AI Battery Fast Charging</h1>
    <p class="hero-sub">Soft Actor-Critic reinforcement learning for lithium-ion cell management</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# state: keep last simulation in session
# ──────────────────────────────────────────────────────────────────────
if "hist" not in st.session_state:
    st.session_state.hist = None
    st.session_state.sim_mode = ""

if run_btn:
    with st.spinner("Running simulation …"):
        time.sleep(0.25)   # tiny delay so spinner shows
        st.session_state.hist = run_episode(c_rate, use_rl, rl_model)
        st.session_state.sim_mode = "RL Agent" if use_rl else f"Manual {c_rate:.1f}C"

# ──────────────────────────────────────────────────────────────────────
# tabs
# ──────────────────────────────────────────────────────────────────────
tab_sim, tab_analysis, tab_physics, tab_about = st.tabs([
    "🔋  Live Simulation",
    "📊  Analysis",
    "🧪  Battery Model",
    "ℹ️  About",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ════════════════════════════════════════════════════════════════════
with tab_sim:

    hist = st.session_state.hist

    if hist is None:
        st.markdown("""
        <div class="callout" style="margin-top:30px;font-size:0.93rem;">
        Press <strong>▶ Run Simulation</strong> in the sidebar to start.<br><br>
        Choose <em>RL Agent</em> to watch the trained policy control charging,
        or <em>Manual</em> to apply a fixed C-rate and see how the cell responds.
        </div>
        """, unsafe_allow_html=True)
    else:
        steps = list(range(len(hist["soc"])))
        final_soc  = hist["soc"][-1]
        final_v    = hist["voltage"][-1]
        final_t    = hist["temperature"][-1]
        final_soh  = hist["soh"][-1]
        duration_m = len(steps) * 60 / 60   # minutes

        # ── mode + summary bar ──────────────────
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown(
                f'<span class="pill pill-ok">{st.session_state.sim_mode}</span>'
                f'&nbsp;&nbsp;<span style="color:#6e7681;font-size:0.83rem;">'
                f'{len(steps)} steps · {duration_m:.0f} min simulated</span>',
                unsafe_allow_html=True,
            )
        with col_r:
            overall_ok = final_t < 45 and final_v < 4.20 and final_soh > 0.85
            badge = '<span class="pill pill-ok">Charging safe</span>' if overall_ok \
                else '<span class="pill pill-warn">Limits approached</span>'
            st.markdown(badge, unsafe_allow_html=True)

        # ── SOC progress bar ────────────────────
        st.markdown(soc_bar(final_soc), unsafe_allow_html=True)

        # ── KPI cards ──────────────────────────
        st.markdown(
            '<div class="kpi-row">'
            + kpi_html("Final SOC", f"{final_soc*100:.1f}", "%",
                       "Target: 80 %", "g")
            + kpi_html("Terminal Voltage", f"{final_v:.3f}", "V",
                       "Limit: 4.20 V", "b")
            + kpi_html("Temperature", f"{final_t:.1f}", "°C",
                       f"Safe: < 45 °C  {status_pill(final_t, 35, 43)}",
                       "a")
            + kpi_html("SOH", f"{final_soh*100:.2f}", "%",
                       "Retained capacity", "p")
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── charts  (2 × 2 grid, then reward) ──
        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                line_chart(steps, hist["soc"], "State of Charge", "SOC",
                           "#39d353", threshold=0.80, threshold_label="Target 80 %"),
                width='stretch',
            )
            st.plotly_chart(
                line_chart(steps, hist["temperature"], "Temperature", "°C",
                           "#ffa657", threshold=45, threshold_label="Safety limit"),
                width='stretch',
            )

        with c2:
            st.plotly_chart(
                line_chart(steps, hist["voltage"], "Terminal Voltage", "V",
                           "#58a6ff", threshold=4.20, threshold_label="CV cutoff"),
                width='stretch',
            )
            st.plotly_chart(
                line_chart(steps, hist["soh"], "State of Health", "SOH",
                           "#bc8cff"),
                width='stretch',
            )

        st.plotly_chart(
            line_chart(steps, hist["current"], "Charging Current", "C-rate",
                       "#ff7b72"),
            width='stretch',
        )

        # ── reward curve ───────────────────────
        with st.expander("Show cumulative reward trace"):
            cum_reward = np.cumsum(hist["reward"]).tolist()
            fig_r = line_chart(steps, cum_reward, "Cumulative Reward", "Reward",
                               "#d2a8ff")
            st.plotly_chart(fig_r, width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS  (static plots from training + comparison)
# ════════════════════════════════════════════════════════════════════
with tab_analysis:

    st.markdown("#### Training result plots")
    st.markdown(
        '<p style="color:#6e7681;font-size:0.85rem;">'
        'Generated by running the trained SAC agent at a fixed 1.7 C-rate.</p>',
        unsafe_allow_html=True,
    )

    plots_dir = os.path.join(ROOT, "results", "plots")
    plot_files = {
        "SOC vs Time":         "soc_vs_time.png",
        "Voltage vs Time":     "voltage_vs_time.png",
        "Temperature vs Time": "temperature_vs_time.png",
        "SOH vs Time":         "soh_vs_time.png",
        "Current vs Time":     "current_vs_time.png",
    }

    available = {k: v for k, v in plot_files.items()
                 if os.path.exists(os.path.join(plots_dir, v))}

    if not available:
        st.markdown(
            '<div class="callout">No plots found. Run '
            '<code>python results/generate_plots.py</code> from the project root.</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(2)
        for i, (title, fname) in enumerate(available.items()):
            with cols[i % 2]:
                st.markdown(f'<div class="sec-title">{title}</div>',
                            unsafe_allow_html=True)
                st.image(os.path.join(plots_dir, fname))

    # ── comparison: RL vs manual ─────────────
    if st.session_state.hist is not None:
        st.markdown("---")
        st.markdown("#### RL Agent vs Manual charging — side-by-side")
        st.markdown(
            '<p style="color:#6e7681;font-size:0.85rem;">Run the simulation in both modes '
            'and compare the stored results.</p>',
            unsafe_allow_html=True,
        )

        hist = st.session_state.hist
        steps = list(range(len(hist["soc"])))

        fig_cmp = make_subplots(
            rows=2, cols=2,
            subplot_titles=("SOC", "Voltage", "Temperature", "SOH"),
            vertical_spacing=0.14,
            horizontal_spacing=0.1,
        )
        pairs = [
            (hist["soc"],         "#39d353", 1, 1),
            (hist["voltage"],     "#58a6ff", 1, 2),
            (hist["temperature"], "#ffa657", 2, 1),
            (hist["soh"],         "#bc8cff", 2, 2),
        ]
        for y_data, color, row, col in pairs:
            fig_cmp.add_trace(
                go.Scatter(x=steps, y=y_data, mode="lines",
                           line=dict(color=color, width=2),
                           showlegend=False),
                row=row, col=col,
            )
        fig_cmp.update_layout(height=480, **PLOT_LAYOUT)
        fig_cmp.update_xaxes(gridcolor="#1c2128", tickfont=dict(color="#6e7681"))
        fig_cmp.update_yaxes(gridcolor="#1c2128", tickfont=dict(color="#6e7681"))
        for ann in fig_cmp.layout.annotations:
            ann.font.color = "#8b949e"
            ann.font.size = 12
        st.plotly_chart(fig_cmp, width='stretch')


# ════════════════════════════════════════════════════════════════════
# TAB 3 — BATTERY MODEL
# ════════════════════════════════════════════════════════════════════
with tab_physics:

    st.markdown("#### Single Particle Model — how the physics works")
    st.markdown(
        '<div class="callout">'
        'The environment uses a <strong>simplified Single Particle Model (SPM)</strong> '
        'derived from the Doyle-Fuller-Newman framework. Each electrode is represented by '
        'a single spherical particle with a uniform reaction rate — fast enough for RL '
        'training while retaining key electrochemical relationships.'
        '</div>',
        unsafe_allow_html=True,
    )

    # OCV curve preview
    st.markdown("##### Open-Circuit Voltage (OCV) — NMC 622")
    from battery_model.spm_model import SingleParticleModel, _ocv_nmc

    soc_range = np.linspace(0.01, 0.99, 120)
    ocv_vals  = [_ocv_nmc(s) for s in soc_range]

    fig_ocv = go.Figure()
    fig_ocv.add_trace(go.Scatter(
        x=soc_range, y=ocv_vals,
        mode="lines",
        line=dict(color="#58a6ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        name="OCV (NMC 622)",
        hovertemplate="SOC: %{x:.2f}<br>OCV: %{y:.3f} V<extra></extra>",
    ))
    fig_ocv.update_layout(
        xaxis=_axis("State of Charge"),
        yaxis=_axis("Open-Circuit Voltage (V)"),
        height=280, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_ocv, width='stretch')

    # Internal resistance vs temperature
    st.markdown("##### Internal Resistance vs Temperature")
    temps   = np.linspace(10, 55, 80)
    spm_ref = SingleParticleModel()
    r_vals  = [spm_ref.r_internal(t) for t in temps]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=temps, y=[rv * 1000 for rv in r_vals],
        mode="lines",
        line=dict(color="#ffa657", width=2.5),
        hovertemplate="T: %{x:.0f} °C<br>R_int: %{y:.1f} mΩ<extra></extra>",
    ))
    fig_r.update_layout(
        xaxis=_axis("Temperature (°C)"),
        yaxis=_axis("R_int (mΩ)"),
        height=240, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_r, width='stretch')

    # parameter table
    st.markdown("##### Model parameters")
    st.markdown("""
    <table class="phys-table">
      <thead>
        <tr><th>Parameter</th><th>Symbol</th><th>Value</th><th>Notes</th></tr>
      </thead>
      <tbody>
        <tr><td>Cell chemistry</td><td>—</td><td>NMC 622 / graphite</td><td>Chen 2020 parameterisation</td></tr>
        <tr><td>Nominal capacity</td><td>C_nom</td><td>3.0 Ah</td><td>18650 format</td></tr>
        <tr><td>Internal resistance (25 °C)</td><td>R_int</td><td>48 mΩ</td><td>lumped DC resistance</td></tr>
        <tr><td>R temp. coefficient</td><td>dR/dT</td><td>−0.5 mΩ/°C</td><td>linear fit</td></tr>
        <tr><td>OCV model</td><td>V_OCV(SOC)</td><td>degree-6 polynomial</td><td>fitted to half-cell data</td></tr>
        <tr><td>Overpotential</td><td>η</td><td>Butler-Volmer (linearised)</td><td>α = 0.5</td></tr>
        <tr><td>Thermal model</td><td>—</td><td>I²R heating + convection</td><td>R_th = 0.2 Ω, h = 0.05</td></tr>
        <tr><td>SOH model</td><td>—</td><td>semi-empirical fade</td><td>cycle + calendar stress</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("##### Governing equations")
    st.latex(r"V_t = V_\mathrm{OCV}(\mathrm{SOC}) + i \cdot R_\mathrm{int}(T) + \eta(i,T)")
    st.latex(r"\eta = \frac{RT}{\alpha F}\,\sinh^{-1}\!\left(\frac{i}{2\,i_0(T)}\right)")
    st.latex(r"\mathrm{SOC}(t) = \mathrm{SOC}_0 + \frac{1}{C_\mathrm{nom}}\int_0^t i(\tau)\,d\tau")
    st.latex(r"\dot{T} = \frac{i^2 R_\mathrm{th} - h(T - T_\mathrm{amb})}{m c_p}")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
with tab_about:

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="about-card">
            <h3>🔋 What is this?</h3>
            <p>
            This project trains a <strong>Soft Actor-Critic (SAC)</strong> reinforcement
            learning agent to control lithium-ion battery charging. The agent learns to
            apply the right current at each second — charging fast while keeping the cell
            safe and healthy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🤖 The RL Agent</h3>
            <p>
            SAC is an off-policy, entropy-regularised algorithm. It simultaneously maximises
            expected reward and policy entropy, which makes it naturally robust to noisy
            observations — ideal for real battery systems where sensors drift over time.<br><br>
            Trained for 20 000 environment steps (~14 days of simulated charging).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-card">
            <h3>⚡ Why does this matter?</h3>
            <p>
            Conventional CC-CV charging is conservative by design. An adaptive AI policy can
            charge 10–25 % faster in simulation while applying less cumulative stress,
            because it actively trades off speed against heat and aging at every step.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🔬 Tech stack</h3>
            <p>
            <strong>Python</strong> · <strong>Stable-Baselines3</strong> (SAC) ·
            <strong>Gymnasium</strong> (custom env) · <strong>PyBaMM</strong> (electrochemistry) ·
            <strong>Streamlit + Plotly</strong> (this dashboard) ·
            <strong>PyTorch</strong> (neural network backend)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="about-card">
        <h3>🗺️ Roadmap</h3>
        <p>
        • Full PyBaMM state coupling (solid-phase diffusion, electrolyte concentration)<br>
        • Battery aging dataset integration (NASA, Oxford) for data-driven SOH<br>
        • Multi-cell pack simulation with thermal interaction<br>
        • Digital Twin export for hardware-in-the-loop BMS testing<br>
        • Real-time deployment via CAN bus interface
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center;color:#484f58;font-size:0.78rem;margin-top:28px;">
    AI Battery Fast Charging Optimizer · SAC + SPM · Built with Streamlit
    </p>
    """, unsafe_allow_html=True)
