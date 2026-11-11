import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.linalg import expm

from src.simulation import compare, simulate
from src.scenarios import SCENARIOS
from src.thermal_model import linearize
from src.stability import classify, root_locus
from src.frequency_domain import bode
from src.time_domain import response, specs
from src.state_space import ctrb, obsv, cayley_hamilton_expm
from src.block_diagrams import series, parallel, feedback
from src.signal_flow import mason
from src.electrical_models import rlc
from src.mechanical_models import mass_spring_damper
from src.compensators import lead, lag, lead_lag, controller
from config.default_parameters import CAPACITANCE, NATURAL_H, COOLING_BETA, COUPLING, GAMMA

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title='ThermoGuard | BESS Thermal Management & Control Laboratory',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom High-End Styling
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #0d192e 0%, #060b13 100%);
    color: #e2e8f0;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09111e 0%, #060b14 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.12);
}
section[data-testid="stSidebar"] .stRadio label {
    font-weight: 500;
    color: #94a3b8;
}

/* Glassmorphic Cards */
.tg-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.tg-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
}

.tg-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 20px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.tg-badge-cyan { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
.tg-badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.tg-badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
.tg-badge-rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}
div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    color: #f8fafc;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #94a3b8;
    font-weight: 500;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.08) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 8px;
    padding: 8px 22px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    border-color: rgba(56, 189, 248, 0.8);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
}

/* Dataframe & Tables */
.stDataFrame {
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    overflow: hidden;
}

/* Code & LaTeX */
.katex-display {
    padding: 12px;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.1);
}
</style>
''', unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def make_dark_figure(title="", height=380, xlabel="", ylabel=""):
    """Create a dark glassmorphic styled Plotly figure."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(family='Outfit', size=16, color='#f1f5f9')),
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.45)',
        xaxis=dict(
            title=xlabel,
            gridcolor='rgba(255, 255, 255, 0.07)',
            zerolinecolor='rgba(255, 255, 255, 0.15)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            title=ylabel,
            gridcolor='rgba(255, 255, 255, 0.07)',
            zerolinecolor='rgba(255, 255, 255, 0.15)',
            tickfont=dict(color='#94a3b8')
        ),
        legend=dict(
            bgcolor='rgba(15, 23, 42, 0.8)',
            bordercolor='rgba(56, 189, 248, 0.2)',
            borderwidth=1,
            font=dict(color='#cbd5e1', size=11)
        ),
        margin=dict(l=50, r=30, t=50, b=45)
    )
    return fig

def matrix_to_latex(mat, decimals=4):
    """Format a 2D numpy matrix into clean LaTeX string."""
    lines = []
    for row in mat:
        formatted_row = []
        for x in row:
            if abs(x) < 1e-12:
                formatted_row.append("0")
            elif abs(x) >= 1e-3 and abs(x) < 1e4:
                formatted_row.append(f"{x:.{decimals}f}")
            else:
                formatted_row.append(f"{x:.2e}")
        lines.append(" & ".join(formatted_row))
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

ZONE_COLORS = ['#00f2fe', '#a855f7', '#fbbf24', '#34d399']

# ---------------------------------------------------------
# Sidebar Navigation & Brand
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('''
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <div style="background:linear-gradient(135deg, #0ea5e9, #6366f1); border-radius:10px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-size:20px; box-shadow:0 4px 12px rgba(14,165,233,0.3);">⚡</div>
        <div>
            <div style="font-family:'Outfit'; font-weight:700; font-size:20px; color:#f8fafc; line-height:1.2;">ThermoGuard</div>
            <div style="font-size:11px; color:#38bdf8; letter-spacing:0.04em;">BESS CONTROL LABORATORY</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    workspace = st.radio(
        'Workspace',
        ['BESS Digital Simulator', 'Control Systems Laboratory', 'Project Information & Syllabus'],
        label_visibility='collapsed'
    )
    
    st.markdown('---')
    st.markdown('''
    <div style="background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.15); border-radius:8px; padding:12px; font-size:12px; color:#94a3b8;">
        <span class="tg-badge tg-badge-cyan">Engine Active</span><br/>
        <div style="margin-top:6px;">• 4 Coupled Thermal Zones (RK4)</div>
        <div>• SciPy LTI SISO/MIMO Analysis</div>
        <div>• Real-time Allocation Heuristics</div>
    </div>
    ''', unsafe_allow_html=True)

# =========================================================
# WORKSPACE 1: BESS DIGITAL SIMULATOR
# =========================================================
if workspace == 'BESS Digital Simulator':
    st.markdown('''
    <div style="margin-bottom: 20px;">
        <h1 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">BESS Thermal Management Simulator</h1>
        <p style="color:#94a3b8; font-size:14px; margin:0;">
            Real-time numerical simulation of 4 thermally coupled battery pack zones with cooling budget constraints and fault injection.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Preset configurations
    pcol1, pcol2, pcol3, pcol4 = st.columns([2, 1, 1, 1])
    with pcol1:
        scenario = st.selectbox('Select Operating Scenario', list(SCENARIOS), index=0)
    
    # Preset quick-fill triggers
    if 'preset_loaded' not in st.session_state:
        st.session_state.preset_loaded = False
        
    st.info(f"**Scenario Profile:** {SCENARIOS[scenario]}")
    
    # Parameter Controls Grouped in Expanders
    with st.expander('⚙️ Simulation & Physical Parameters', expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('**🕒 Runtime & Environment**')
            duration = st.slider('Simulation Duration (s)', 300, 3600, 1800, 100)
            dt = st.select_slider(r'Integrator Step $\Delta t$ (s)', options=[1.0, 2.0, 5.0, 10.0], value=2.0)
            ambient_c = st.slider(r'Ambient Temperature $T_{amb}$ (°C)', 15.0, 45.0, 25.0, 0.5)
            ambient = ambient_c + 273.15
        with col2:
            st.markdown('**🔋 Pack Load & Control Limits**')
            current = st.slider(r'Pack Current Magnitude $I$ (A)', 50.0, 400.0, 180.0, 10.0)
            setpoint_c = st.slider(r'Temperature Target $T_{sp}$ (°C)', 20.0, 40.0, 30.0, 0.5)
            setpoint = setpoint_c + 273.15
            budget = st.slider(r'Total Cooling Budget $\sum u_i$', 0.0, 4.0, 2.0, 0.1)
        with col3:
            st.markdown('**⚠️ Disturbance & Injected Faults**')
            heat = st.slider(r'Hotspot Heat Pulse $Q_{fault}$ (W)', 0.0, 50000.0, 18000.0, 1000.0)
            onset = st.slider(r'Fault Onset Time $t_{on}$ (s)', 0.0, float(duration), 300.0, 50.0)
            health = st.slider(r'Degraded Actuator Health $\eta$', 0.05, 1.0, 0.45, 0.05)
            bias = st.slider(r'Zone 1 Sensor Bias $\Delta T$ (K)', -10.0, 10.0, 2.0, 0.5)
            noise = st.slider(r'Sensor Noise $\sigma$ (K)', 0.0, 1.0, 0.0, 0.05)
            fixed = st.slider(r'Open-Loop Fixed Command $u_{fixed}$', 0.0, 1.0, 0.35, 0.05)
            
    # Run Simulation Execution
    run_clicked = st.button('🚀 Run Simulation', type='primary', use_container_width=True)
    if run_clicked or 'bess_results' not in st.session_state:
        kwargs = dict(
            scenario=scenario, duration=duration, dt=dt, ambient=ambient,
            current=current, setpoint=setpoint, budget=budget, heat_fault=heat,
            fault_on=onset, health_fault=health, sensor_bias=bias,
            sensor_noise=noise, fixed=fixed
        )
        try:
            with st.spinner('Integrating coupled 4-zone thermal dynamics...'):
                st.session_state.bess_results = compare(**kwargs)
                st.session_state.bess_kwargs = kwargs
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.stop()
            
    results = st.session_state.bess_results
    
    # Controller Selection for Detailed View
    d_col1, d_col2 = st.columns([1, 2])
    with d_col1:
        selected_ctrl = st.selectbox('Active Controller Telemetry', ['ThermoGuard', 'PI/PID', 'Open-loop'])
    with d_col2:
        if selected_ctrl == 'ThermoGuard':
            st.markdown('''<span class="tg-badge tg-badge-cyan">Active Controller: ThermoGuard Heuristic</span> 
            <span style="font-size:12px; color:#94a3b8; margin-left:8px;">Dynamically allocates cooling based on temperature margin, rise rate & heat feedforward.</span>''', unsafe_allow_html=True)
        elif selected_ctrl == 'PI/PID':
            st.markdown('''<span class="tg-badge tg-badge-amber">Active Controller: Average PI/PID</span> 
            <span style="font-size:12px; color:#94a3b8; margin-left:8px;">SISO PI feedback based on average measured temperature; equal command to all zones.</span>''', unsafe_allow_html=True)
        else:
            st.markdown('''<span class="tg-badge tg-badge-rose">Active Controller: Open-Loop Fixed</span> 
            <span style="font-size:12px; color:#94a3b8; margin-left:8px;">Constant fixed command $u_i = u_{fixed}$ regardless of thermal dynamics.</span>''', unsafe_allow_html=True)

    r = results[selected_ctrl]
    t = r['t']
    T = r['T']
    U = r['U']
    health_mat = r['health']

    # Zone Status Cards
    st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
    zcols = st.columns(4)
    for i, col in enumerate(zcols):
        temp_c = T[-1, i] - 273.15
        rate_k_min = (T[-1, i] - T[-2, i]) / dt * 60.0
        zone_health = health_mat[-1, i]
        cooling_u = U[-1, i]
        
        # Color coding for temperature status
        if temp_c > (setpoint_c + 5.0):
            badge = '<span class="tg-badge tg-badge-rose">Critical Hotspot</span>'
            status_color = '#f43f5e'
        elif temp_c > setpoint_c:
            badge = '<span class="tg-badge tg-badge-amber">Elevated</span>'
            status_color = '#fbbf24'
        else:
            badge = '<span class="tg-badge tg-badge-green">Nominal</span>'
            status_color = '#34d399'

        col.markdown(f'''
        <div class="tg-card" style="border-left: 4px solid {status_color}; padding:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; font-size:15px; color:#f8fafc;">Zone {i+1}</span>
                {badge}
            </div>
            <div style="font-size:24px; font-weight:800; font-family:'Outfit'; color:#ffffff; margin-bottom:4px;">
                {temp_c:.2f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">°C</span>
            </div>
            <div style="font-size:12px; color:{'#34d399' if rate_k_min <= 0 else '#fbbf24'}; margin-bottom:8px;">
                {'↓' if rate_k_min <= 0 else '↑'} {abs(rate_k_min):.2f} K/min
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">
                <span>Cooling: <strong style="color:#38bdf8;">{cooling_u:.2f}</strong></span>
                <span>Actuator: <strong style="color:{'#34d399' if zone_health>0.9 else '#fb7185'};">{zone_health*100:.0f}%</strong></span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Detailed Analysis Tabs
    tabs = st.tabs(['📊 Telemetry & Actuation', '⚖️ Benchmark Comparison', '🔍 Fault & Sensor Diagnostics'])

    # TAB 1: TELEMETRY & ACTUATION
    with tabs[0]:
        c_left, c_right = st.columns(2)
        with c_left:
            fig_temp = make_dark_figure(
                title=f"{selected_ctrl}: Zone Temperatures over Time",
                height=400, xlabel="Time (s)", ylabel="Temperature (°C)"
            )
            for i in range(4):
                fig_temp.add_trace(go.Scatter(
                    x=t, y=T[:, i] - 273.15,
                    mode='lines', name=f'Zone {i+1}',
                    line=dict(color=ZONE_COLORS[i], width=2.2)
                ))
            # Setpoint reference line
            fig_temp.add_hline(
                y=setpoint_c, line_dash='dash', line_color='#94a3b8',
                annotation_text='Setpoint', annotation_position='top right'
            )
            # Threshold line
            fig_temp.add_hline(
                y=setpoint_c + 5.0, line_dash='dot', line_color='#f43f5e',
                annotation_text='Threshold (SP + 5K)', annotation_position='top left'
            )
            # Highlight fault interval if applicable
            if onset < duration and 'hotspot' in scenario.lower():
                fig_temp.add_vrect(
                    x0=onset, x1=min(duration, onset + 600.0),
                    fillcolor="rgba(244, 63, 94, 0.08)", line_width=0,
                    annotation_text="Hotspot Window", annotation_position="top left"
                )
            st.plotly_chart(fig_temp, use_container_width=True)

        with c_right:
            fig_act = make_dark_figure(
                title=f"{selected_ctrl}: Cooling Actuator Commands & Allocation",
                height=400, xlabel="Time (s)", ylabel="Command Effort u (0–1)"
            )
            for i in range(4):
                fig_act.add_trace(go.Scatter(
                    x=t, y=U[:, i],
                    mode='lines', name=f'Zone {i+1} Command',
                    line=dict(color=ZONE_COLORS[i], width=2)
                ))
            # Total allocated budget line
            fig_act.add_trace(go.Scatter(
                x=t, y=U.sum(axis=1),
                mode='lines', name='Total Budget Usage',
                line=dict(color='#f8fafc', width=1.5, dash='dash')
            ))
            fig_act.add_hline(
                y=budget, line_dash='dot', line_color='#38bdf8',
                annotation_text=f'Max Budget ({budget:.1f})', annotation_position='bottom right'
            )
            st.plotly_chart(fig_act, use_container_width=True)

        # CSV Download Section
        df_csv = pd.DataFrame({
            'time_s': t,
            **{f'zone_{i+1}_temp_C': T[:, i] - 273.15 for i in range(4)},
            **{f'command_u_{i+1}': U[:, i] for i in range(4)},
            **{f'actuator_health_{i+1}': health_mat[:, i] for i in range(4)},
            'total_cooling_command': U.sum(axis=1)
        })
        st.download_button(
            label=f'📥 Export {selected_ctrl} Simulation Data (CSV)',
            data=df_csv.to_csv(index=False),
            file_name=f'thermoguard_{selected_ctrl.lower().replace("/", "_")}_run.csv',
            mime='text/csv'
        )

    # TAB 2: BENCHMARK COMPARISON
    with tabs[1]:
        st.markdown('### Comparative Policy Performance')
        st.caption('Direct side-by-side benchmark of Open-Loop, Conventional Average PI/PID, and Zone-Aware ThermoGuard under identical disturbances.')

        # Top summary cards for 3 controllers
        bcols = st.columns(3)
        for col, (name, x) in zip(bcols, results.items()):
            m = x['metrics']
            peak_temp = m['max_temperature_C']
            energy_proxy = m['cooling_energy_proxy_s']
            max_spread = m['max_spread_K']
            budget_util = m['max_budget_utilization']

            highlight = (name == 'ThermoGuard')
            badge_html = '<span class="tg-badge tg-badge-cyan">Recommended</span>' if highlight else ''
            card_border = 'rgba(14, 165, 233, 0.6)' if highlight else 'rgba(255,255,255,0.08)'
            card_bg = 'rgba(14, 165, 233, 0.08)' if highlight else 'rgba(15, 23, 42, 0.6)'
            budget_color = '#34d399' if budget_util <= budget + 0.01 else '#f43f5e'
            card_html = (
                f'<div class="tg-card" style="border: 1px solid {card_border}; background: {card_bg};">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">'
                f'<span style="font-weight:700; font-size:16px; color:#ffffff;">{name}</span>'
                f'{badge_html}'
                f'</div>'
                f'<div style="font-size:26px; font-weight:800; font-family:\'Outfit\'; color:#ffffff; margin-bottom:4px;">'
                f'{peak_temp:.2f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">°C peak</span>'
                f'</div>'
                f'<div style="font-size:12px; color:#cbd5e1; margin-bottom:6px;">'
                f'Spatial Spread: <strong style="color:#fbbf24;">{max_spread:.2f} K</strong>'
                f'</div>'
                f'<div style="font-size:12px; color:#cbd5e1; margin-bottom:6px;">'
                f'Fan Energy Proxy: <strong style="color:#38bdf8;">{energy_proxy:.1f}</strong>'
                f'</div>'
                f'<div style="font-size:12px; color:#cbd5e1;">'
                f'Budget Utilization: <strong style="color:{budget_color};">{budget_util:.2f} / {budget:.1f}</strong>'
                f'</div>'
                f'</div>'
            )
            col.markdown(card_html, unsafe_allow_html=True)

        # Comparative Plots
        cmp_left, cmp_right = st.columns(2)
        with cmp_left:
            fig_cmp = make_dark_figure(
                title="Hottest Zone Temperature Over Time",
                height=380, xlabel="Time (s)", ylabel="Max Zone Temperature (°C)"
            )
            ctrl_palette = {'Open-loop': '#f43f5e', 'PI/PID': '#fbbf24', 'ThermoGuard': '#00f2fe'}
            for name, x in results.items():
                fig_cmp.add_trace(go.Scatter(
                    x=x['t'], y=x['T'].max(axis=1) - 273.15,
                    mode='lines', name=name,
                    line=dict(color=ctrl_palette.get(name, '#ffffff'), width=2.5 if name == 'ThermoGuard' else 1.8)
                ))
            fig_cmp.add_hline(y=setpoint_c, line_dash='dash', line_color='#94a3b8', annotation_text='Setpoint')
            st.plotly_chart(fig_cmp, use_container_width=True)

        with cmp_right:
            # Energy & Budget Bar Chart
            fig_bar = make_dark_figure(
                title="Cooling Energy Proxy (∫Σu³dt) Comparison",
                height=380, xlabel="Control Policy", ylabel="Energy Proxy (s)"
            )
            ctrl_names = list(results.keys())
            energies = [results[n]['metrics']['cooling_energy_proxy_s'] for n in ctrl_names]
            bar_colors = [ctrl_palette.get(n, '#0ea5e9') for n in ctrl_names]
            fig_bar.add_trace(go.Bar(
                x=ctrl_names, y=energies,
                marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
                text=[f"{v:.1f}" for v in energies], textposition='auto'
            ))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Full Metrics Dataframe
        st.markdown('#### Detailed Benchmark Metrics')
        metrics_dict = {}
        for n, x in results.items():
            m = x['metrics']
            metrics_dict[n] = {
                'Peak Temp (°C)': f"{m['max_temperature_C']:.2f}",
                'Max Spatial Spread (K)': f"{m['max_spread_K']:.2f}",
                'IAE Tracking Error (K·s)': f"{m['IAE_Ks']:.1f}",
                'Time Above SP+5K (s)': f"{m['time_above_threshold_s']:.1f}",
                'Cooling Energy Proxy (s)': f"{m['cooling_energy_proxy_s']:.1f}",
                'Peak Actuator Command': f"{m['peak_command']:.2f}",
                'Max Budget Usage': f"{m['max_budget_utilization']:.2f}",
                'Final Steady Error (K)': f"{m['steady_state_error_K']:.2f}"
            }
        df_metrics = pd.DataFrame(metrics_dict).T
        st.dataframe(df_metrics, use_container_width=True)

    # TAB 3: FAULT & SENSOR DIAGNOSTICS
    with tabs[2]:
        st.markdown('### Sensor Fidelity & Physical Disturbance Analysis')
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            fig_sensor = make_dark_figure(
                title="Zone 1: Ground Truth vs Measured Telemetry",
                height=380, xlabel="Time (s)", ylabel="Temperature (°C)"
            )
            fig_sensor.add_trace(go.Scatter(
                x=t, y=T[:, 0] - 273.15,
                mode='lines', name='True Physical State',
                line=dict(color='#00f2fe', width=2.5)
            ))
            fig_sensor.add_trace(go.Scatter(
                x=t, y=r['measured'][:, 0] - 273.15,
                mode='lines', name='Noisy/Biased Sensor Reading',
                line=dict(color='#fb7185', width=1.5, dash='dash')
            ))
            st.plotly_chart(fig_sensor, use_container_width=True)

        with fcol2:
            fig_heat = make_dark_figure(
                title="Injected Heat Disturbance Profiles Q(t)",
                height=380, xlabel="Time (s)", ylabel="Heat Generation (W)"
            )
            heat_mat = r['heat']
            for i in range(4):
                fig_heat.add_trace(go.Scatter(
                    x=t, y=heat_mat[:, i],
                    mode='lines', name=f'Zone {i+1} Heat',
                    line=dict(color=ZONE_COLORS[i], width=2)
                ))
            st.plotly_chart(fig_heat, use_container_width=True)

        # Thermal Model Physics Explainer
        st.markdown('''
        <div class="tg-card">
            <h4 style="color:#f8fafc; margin-top:0;">Coupled Multi-Zone Thermal Physics Model</h4>
            <div style="font-size:14px; color:#cbd5e1; line-height:1.6;">
                Each battery zone temperature $T_i$ follows the non-linear differential balance:
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r'C_i \frac{dT_i}{dt} = Q_i - (h_{0i} + \beta_i \eta_i u_i)(T_i - T_a) + \sum_{j \neq i} k_{ij}(T_j - T_i)')
        
        st.markdown(f'''
        <div style="font-size:13px; color:#94a3b8; display:flex; gap:20px; flex-wrap:wrap; margin-top:10px;">
            <div>• <strong>Capacitance $C_i$:</strong> {CAPACITANCE[0]:.1e} to {CAPACITANCE[3]:.1e} J/K</div>
            <div>• <strong>Natural Convection $h_{{0i}}$:</strong> {NATURAL_H.mean():.0f} W/K</div>
            <div>• <strong>Forced Cooling $\\beta_i$:</strong> {COOLING_BETA.mean():.0f} W/K</div>
            <div>• <strong>Inter-zone Conductance $k_{{ij}}$:</strong> 65–70 W/K</div>
        </div>
        ''', unsafe_allow_html=True)

# =========================================================
# WORKSPACE 2: CONTROL SYSTEMS LABORATORY
# =========================================================
elif workspace == 'Control Systems Laboratory':
    st.markdown('''
    <div style="margin-bottom: 20px;">
        <h1 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">Control Systems Laboratory</h1>
        <p style="color:#94a3b8; font-size:14px; margin:0;">
            Interactive numerical demonstrations spanning classical SISO feedback to modern state-space BESS control.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    module = st.selectbox(
        'Syllabus Module',
        [
            'Module 1 · Feedback Principles & Block Diagrams',
            'Module 2 · Mathematical Modelling & Analogy',
            'Module 3 · Time-Domain Transient Response',
            'Module 4 · Stability Analysis & Root Locus',
            'Module 5 · Frequency-Domain Analysis (Bode & Nyquist)',
            'Module 6 · Classical Controllers & Compensators',
            'Module 7 · State-Space & Modern Control Analysis'
        ]
    )

    # -----------------------------------------------------
    # MODULE 1: Principles of Feedback & Block Diagrams
    # -----------------------------------------------------
    if module.startswith('Module 1'):
        st.subheader('Module 1 · Feedback Principles & Closed-Loop Reduction')
        st.markdown('''
        Negative feedback closes the loop around a plant $G(s)$ with feedback sensor dynamics $H(s)$.
        Feedback desensitizes the loop to plant parameter variations, improves bandwidth, and suppresses input disturbances.
        ''')
        st.latex(r'T(s) = \frac{Y(s)}{R(s)} = \frac{G(s)}{1 + G(s)H(s)}')

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('**Loop Parameters**')
            K_plant = st.slider('Forward Plant Gain $K$', 0.1, 10.0, 2.0, 0.1)
            H_sensor = st.slider('Feedback Sensor Gain $H$', 0.1, 5.0, 1.0, 0.1)
            tau_plant = st.slider('Plant Time Constant $\\tau$ (s)', 0.2, 5.0, 1.0, 0.1)
            
            # Theoretical metrics
            cl_gain = K_plant / (1.0 + K_plant * H_sensor)
            cl_tau = tau_plant / (1.0 + K_plant * H_sensor)
            st.markdown(f'''
            <div class="tg-card" style="padding:12px; margin-top:14px;">
                <div style="font-size:12px; color:#94a3b8;">Closed-Loop DC Gain:</div>
                <div style="font-size:18px; font-weight:700; color:#38bdf8;">{cl_gain:.3f}</div>
                <div style="font-size:12px; color:#94a3b8; margin-top:8px;">Closed-Loop Time Constant:</div>
                <div style="font-size:18px; font-weight:700; color:#34d399;">{cl_tau:.3f} s</div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            t_grid = np.linspace(0, 15, 600)
            _, y_ol = response([K_plant], [tau_plant, 1.0], t_grid)
            _, y_cl = response([K_plant], [tau_plant, 1.0 + K_plant * H_sensor], t_grid)

            fig_fb = make_dark_figure("Step Response: Open-Loop vs Closed-Loop", height=380, xlabel="Time (s)", ylabel="Output y(t)")
            fig_fb.add_trace(go.Scatter(x=t_grid, y=y_ol, mode='lines', name=f'Open-Loop G(s) [K={K_plant:.1f}]', line=dict(color='#fbbf24', dash='dash')))
            fig_fb.add_trace(go.Scatter(x=t_grid, y=y_cl, mode='lines', name=f'Closed-Loop T(s) [H={H_sensor:.1f}]', line=dict(color='#00f2fe', width=2.5)))
            fig_fb.add_hline(y=1.0, line_dash='dot', line_color='#64748b', annotation_text='Reference Input r(t)=1')
            st.plotly_chart(fig_fb, use_container_width=True)

        st.markdown('---')
        st.markdown('#### Block Diagram Algebra Calculator')
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            g1_s = st.number_input('Series $G_1$', value=2.0)
            g2_s = st.number_input('Series $G_2$', value=3.0)
            st.info(f"**Series Reduction:** $G_{{eq}} = G_1 \\times G_2 = {series(g1_s, g2_s):.2f}$")
        with bcol2:
            g1_p = st.number_input('Parallel $G_1$', value=2.0)
            g2_p = st.number_input('Parallel $G_2$', value=3.0)
            st.info(f"**Parallel Reduction:** $G_{{eq}} = G_1 + G_2 = {parallel(g1_p, g2_p):.2f}$")
        with bcol3:
            g_f = st.number_input('Feedback Forward $G$', value=2.0)
            h_f = st.number_input('Feedback Sensor $H$', value=1.0)
            st.info(f"**Closed-Loop Reduction:** $T = \\frac{{G}}{{1+GH}} = {feedback(g_f, h_f):.3f}$")

    # -----------------------------------------------------
    # MODULE 2: Mathematical Modelling & Analogy
    # -----------------------------------------------------
    elif module.startswith('Module 2'):
        st.subheader('Module 2 · Mathematical Modelling & Physical Analogies')
        
        # Analogy Table
        st.markdown('''
        #### Thermal – Electrical – Mechanical Dynamic Analogies
        Dynamic systems in different physical domains obey analogous differential equations:
        ''')
        df_analogy = pd.DataFrame({
            'Physical Domain': ['Thermal System', 'Electrical System', 'Mechanical (Translational)'],
            'Across Variable (Potential)': ['Temperature Difference $\Delta T$ (K)', 'Voltage Difference $V$ (V)', 'Velocity $v$ (m/s)'],
            'Through Variable (Flow)': ['Heat Flow Rate $q$ (W)', 'Current $i$ (A)', 'Force $F$ (N)'],
            'Energy Storage (Capacitive)': ['Thermal Capacitance $C_{th}$ (J/K)', 'Capacitance $C$ (F)', 'Mass $m$ (kg)'],
            'Dissipative Element': ['Thermal Resistance $R_{th} = 1/h$ (K/W)', 'Resistance $R$ ($\Omega$)', 'Viscous Friction $c$ (N·s/m)']
        })
        st.dataframe(df_analogy, use_container_width=True, hide_index=True)

        # Linearized BESS Model
        st.markdown('---')
        st.markdown('#### Linearized 4-Zone BESS State-Space Representation')
        st.markdown('Local linearization of the coupled non-linear thermal equations around nominal operating point ($T_{op} = 30^\circ\text{C}, u_{op} = 0.35$):')
        A, B, C, D, E = linearize()
        
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown('**Plant Matrix $\mathbf{A}$:**')
            st.latex(r'\mathbf{A} = ' + matrix_to_latex(A, decimals=4))
        with mcol2:
            st.markdown('**Actuator Input Matrix $\mathbf{B}$:**')
            st.latex(r'\mathbf{B} = ' + matrix_to_latex(B, decimals=4))

        eigvals = np.linalg.eigvals(A)
        st.markdown(f"**Linearized System Poles (Eigenvalues $\lambda_i$):** {', '.join([f'{e:.4f} s⁻¹' for e in eigvals])}")

        # Mass-Spring-Damper & Series RLC Interactive
        st.markdown('---')
        st.markdown('#### Dual Physical System Demonstrators')
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown('**Mechanical Mass-Spring-Damper System**')
            st.latex(r'm \ddot{x} + c \dot{x} + k x = F(t) \implies G(s) = \frac{1}{m s^2 + c s + k}')
            m_val = st.slider('Mass $m$ (kg)', 0.5, 5.0, 1.0, 0.5)
            c_val = st.slider('Damping $c$ (N·s/m)', 0.2, 5.0, 2.0, 0.2)
            k_val = st.slider('Stiffness $k$ (N/m)', 1.0, 25.0, 10.0, 1.0)
            
            sys_m = mass_spring_damper(m=m_val, c=c_val, k=k_val)
            t_m = np.linspace(0, 10, 500)
            _, y_m = response(sys_m.num, sys_m.den, t_m)
            fig_msd = make_dark_figure("Mass-Spring-Damper Step Displacement", height=300, xlabel="Time (s)", ylabel="Displacement x(t)")
            fig_msd.add_trace(go.Scatter(x=t_m, y=y_m, mode='lines', line=dict(color='#34d399', width=2)))
            st.plotly_chart(fig_msd, use_container_width=True)

        with pcol2:
            st.markdown('**Electrical Series RLC Circuit**')
            st.latex(r'L C \frac{d^2 v_c}{dt^2} + R C \frac{d v_c}{dt} + v_c = v_{in} \implies G(s) = \frac{1}{L C s^2 + R C s + 1}')
            r_val = st.slider('Resistance $R$ ($\Omega$)', 1.0, 50.0, 10.0, 1.0)
            l_val = st.slider('Inductance $L$ (H)', 0.01, 1.0, 0.1, 0.01)
            c_elec = st.slider('Capacitance $C$ (mF)', 0.1, 5.0, 1.0, 0.1) * 1e-3
            
            sys_e = rlc(R=r_val, L=l_val, C=c_elec)
            t_e = np.linspace(0, 0.2, 500)
            _, y_e = response(sys_e.num, sys_e.den, t_e)
            fig_rlc = make_dark_figure("Capacitor Voltage Step Response", height=300, xlabel="Time (s)", ylabel="Voltage vc(t)")
            fig_rlc.add_trace(go.Scatter(x=t_e, y=y_e, mode='lines', line=dict(color='#fbbf24', width=2)))
            st.plotly_chart(fig_rlc, use_container_width=True)

        # Mason's Gain Formula
        st.markdown('---')
        st.markdown("#### Mason's Gain Formula for Signal Flow Graphs")
        st.latex(r'T = \frac{1}{\Delta} \sum_k P_k \Delta_k')
        st.markdown('''
        *Example Topology:* Forward path $P_1 = 2$ touching nodes $\{1,2,3\}$. Feedback loops $L_1 = -0.2$ (nodes $\{2,3\}$) and $L_2 = -0.1$ (node $\{4\}$).
        Loops $L_1$ and $L_2$ do not touch, and $L_2$ does not touch path $P_1$.
        ''')
        res_m = mason([(2, {1, 2, 3})], [(-0.2, {2, 3}), (-0.1, {4})])
        st.json(res_m)

    # -----------------------------------------------------
    # MODULE 3: Time-Domain Transient Response
    # -----------------------------------------------------
    elif module.startswith('Module 3'):
        st.subheader('Module 3 · Time-Domain Transient Response & Performance Specifications')
        
        tcol1, tcol2 = st.columns([1, 2])
        with tcol1:
            kind_input = st.selectbox('Test Input Signal', ['step', 'ramp', 'impulse'])
            order = st.radio('System Order', ['Second order', 'First order'])
            K_gain = st.slider('Steady-State Gain $K$', 0.1, 10.0, 1.0, 0.1)
            
            if order == 'First order':
                tau_val = st.slider('Time Constant $\\tau$ (s)', 0.1, 10.0, 2.0, 0.1)
                num = [K_gain]
                den = [tau_val, 1.0]
                st.latex(r'G(s) = \frac{K}{\tau s + 1}')
            else:
                wn_val = st.slider('Natural Frequency $\omega_n$ (rad/s)', 0.2, 5.0, 1.0, 0.1)
                z_val = st.slider('Damping Ratio $\zeta$', 0.05, 2.0, 0.7, 0.05)
                num = [K_gain * wn_val**2]
                den = [1.0, 2.0 * z_val * wn_val, wn_val**2]
                st.latex(r'G(s) = \frac{K \omega_n^2}{s^2 + 2\zeta \omega_n s + \omega_n^2}')

        with tcol2:
            t_span = np.linspace(0, 30, 1200)
            tx, y_resp = response(num, den, t_span, kind=kind_input)
            
            fig_time = make_dark_figure(f"{order} System {kind_input.capitalize()} Response", height=380, xlabel="Time (s)", ylabel="Output y(t)")
            fig_time.add_trace(go.Scatter(x=tx, y=y_resp, mode='lines', name='Response y(t)', line=dict(color='#00f2fe', width=2.5)))
            
            if kind_input == 'step':
                fig_time.add_hline(y=K_gain, line_dash='dash', line_color='#94a3b8', annotation_text=f'Target (K={K_gain})')
                # 2% settling bounds
                fig_time.add_hrect(y0=0.98 * K_gain, y1=1.02 * K_gain, fillcolor="rgba(52, 211, 153, 0.1)", line_width=0, annotation_text="±2% Band")
            st.plotly_chart(fig_time, use_container_width=True)

        if kind_input == 'step':
            st.markdown('#### Extracted Transient Specifications')
            sp = specs(tx, y_resp, final=K_gain)
            scols = st.columns(5)
            scols[0].metric('Rise Time $T_r$ (10–90%)', f"{sp['rise_10_90_s']:.2f} s" if np.isfinite(sp['rise_10_90_s']) else "N/A")
            scols[1].metric('Peak Time $T_p$', f"{sp['peak_time_s']:.2f} s")
            scols[2].metric('Overshoot %', f"{sp['overshoot_percent']:.1f}%")
            scols[3].metric('Settling Time $T_s$ (2%)', f"{sp['settling_2pct_s']:.2f} s" if np.isfinite(sp['settling_2pct_s']) else "N/A")
            scols[4].metric('Steady-State Error $e_{ss}$', f"{sp['steady_state_error']:.3f}")

    # -----------------------------------------------------
    # MODULE 4: Stability Analysis & Root Locus
    # -----------------------------------------------------
    elif module.startswith('Module 4'):
        st.subheader('Module 4 · Stability Analysis & Root Locus')
        
        st.markdown('''
        The Routh-Hurwitz criterion determines whether a polynomial has roots in the open right-half of the complex s-plane without explicit factoring.
        Root locus illustrates the trajectory of closed-loop poles as scalar gain $K$ sweeps from $0 \to \infty$.
        ''')

        sc1, sc2 = st.columns([1, 2])
        with sc1:
            a_param = st.slider('Damping Coefficient $a$', -2.0, 4.0, 1.2, 0.1)
            b_param = st.slider('Stiffness Coefficient $b$', -2.0, 4.0, 1.0, 0.1)
            coeff = [1.0, a_param, b_param]
            st.markdown(f"**Characteristic Polynomial:** $P(s) = s^2 + {a_param:.1f}s + {b_param:.1f}$")
            
            res_routh = classify(coeff)
            is_stable = res_routh['stable']
            rhp = res_routh['right_half_plane_poles']
            
            if is_stable:
                st.markdown('<div class="tg-card" style="border-left:4px solid #10b981;"><h4 style="color:#34d399; margin:0;">Status: Asymptotically Stable</h4><div style="font-size:12px; color:#94a3b8; margin-top:4px;">0 sign changes in column 1. All poles lie strictly in LHP.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tg-card" style="border-left:4px solid #f43f5e;"><h4 style="color:#fb7185; margin:0;">Status: Unstable</h4><div style="font-size:12px; color:#94a3b8; margin-top:4px;">{rhp} sign change(s) detected. {rhp} pole(s) in RHP.</div></div>', unsafe_allow_html=True)

            st.markdown('**Routh Array Table:**')
            tab = res_routh['table']
            df_routh = pd.DataFrame(tab, index=[f's^{len(coeff)-1-i}' for i in range(len(tab))], columns=[f'Column {j+1}' for j in range(tab.shape[1])])
            st.dataframe(df_routh, use_container_width=True)

        with sc2:
            gains, poles = root_locus([1.0], [1.0, a_param, b_param])
            fig_rl = make_dark_figure("Closed-Loop Pole Trajectories (Root Locus)", height=420, xlabel="Real Axis $\sigma$", ylabel="Imaginary Axis $j\omega$")
            
            # Scatter pole paths
            fig_rl.add_trace(go.Scatter(
                x=poles.real.flatten(), y=poles.imag.flatten(),
                mode='markers', marker=dict(size=4, color='#00f2fe', opacity=0.7),
                name='Pole Loci as K sweeps'
            ))
            # Open-loop poles (K=0)
            ol_poles = np.roots([1.0, a_param, b_param])
            fig_rl.add_trace(go.Scatter(
                x=ol_poles.real, y=ol_poles.imag,
                mode='markers', marker=dict(size=12, symbol='x', color='#f43f5e'),
                name='Open-Loop Poles (K=0)'
            ))
            # Stability threshold jw axis
            fig_rl.add_vline(x=0, line_dash='dash', line_color='#f8fafc', annotation_text='Stability Boundary')
            fig_rl.add_hline(y=0, line_color='rgba(255,255,255,0.15)')
            st.plotly_chart(fig_rl, use_container_width=True)

    # -----------------------------------------------------
    # MODULE 5: Frequency-Domain Analysis
    # -----------------------------------------------------
    elif module.startswith('Module 5'):
        st.subheader('Module 5 · Frequency-Domain Response (Bode & Nyquist / Polar)')
        
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            wn_bode = st.slider('Natural Frequency $\omega_n$ (rad/s)', 0.1, 10.0, 1.5, 0.1)
            z_bode = st.slider('Damping Ratio $\zeta$', 0.05, 2.0, 0.4, 0.05)
            st.latex(r'G(s) = \frac{\omega_n^2}{s^2 + 2\zeta \omega_n s + \omega_n^2}')
            
            w_bode, mag_bode, phase_bode, margins = bode([wn_bode**2], [1.0, 2.0 * z_bode * wn_bode, wn_bode**2])
            
            pm = margins['phase_margin_deg']
            gm = margins['gain_margin_db']
            w_gc = margins['gain_crossover_rad_s']
            w_pc = margins['phase_crossover_rad_s']
            
            st.markdown(f'''
            <div class="tg-card">
                <h4 style="color:#f8fafc; margin-top:0;">Stability Margins</h4>
                <div style="font-size:13px; color:#94a3b8;">
                    • <strong>Phase Margin:</strong> <span style="color:#34d399; font-weight:700;">{pm:.1f}°</span> at {w_gc:.2f} rad/s<br/>
                    • <strong>Gain Margin:</strong> <span style="color:#38bdf8; font-weight:700;">{gm:.1f} dB</span> at {w_pc if np.isfinite(w_pc) else '∞'}<br/>
                    • <strong>Closed-Loop Status:</strong> {'<span class="tg-badge tg-badge-green">Stable</span>' if pm > 0 and gm > 0 else '<span class="tg-badge tg-badge-rose">Unstable</span>'}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with fcol2:
            # Dual Bode Plots
            fig_mag = make_dark_figure("Bode Magnitude (dB)", height=220, xlabel="Frequency (rad/s)", ylabel="Magnitude (dB)")
            fig_mag.add_trace(go.Scatter(x=w_bode, y=mag_bode, mode='lines', line=dict(color='#00f2fe', width=2)))
            fig_mag.add_hline(y=0, line_dash='dash', line_color='#94a3b8')
            fig_mag.update_xaxes(type='log')
            st.plotly_chart(fig_mag, use_container_width=True)

            fig_phase = make_dark_figure("Bode Phase (deg)", height=220, xlabel="Frequency (rad/s)", ylabel="Phase (deg)")
            fig_phase.add_trace(go.Scatter(x=w_bode, y=phase_bode, mode='lines', line=dict(color='#fbbf24', width=2)))
            fig_phase.add_hline(y=-180, line_dash='dash', line_color='#f43f5e', annotation_text='-180°')
            fig_phase.update_xaxes(type='log')
            st.plotly_chart(fig_phase, use_container_width=True)

        # Polar Nyquist Locus
        st.markdown('#### Polar / Nyquist Locus')
        re_part = np.cos(np.deg2rad(phase_bode)) * 10**(mag_bode / 20.0)
        im_part = np.sin(np.deg2rad(phase_bode)) * 10**(mag_bode / 20.0)
        
        fig_polar = make_dark_figure("Polar Locus in Complex Plane", height=380, xlabel="Real Axis", ylabel="Imaginary Axis")
        fig_polar.add_trace(go.Scatter(x=re_part, y=im_part, mode='lines', name='G(jω)', line=dict(color='#a855f7', width=2.5)))
        # Critical point (-1, 0)
        fig_polar.add_trace(go.Scatter(x=[-1.0], y=[0.0], mode='markers', marker=dict(size=10, symbol='x', color='#f43f5e'), name='Critical Point (-1, j0)'))
        fig_polar.add_hline(y=0, line_color='rgba(255,255,255,0.15)')
        fig_polar.add_vline(x=0, line_color='rgba(255,255,255,0.15)')
        st.plotly_chart(fig_polar, use_container_width=True)

    # -----------------------------------------------------
    # MODULE 6: Classical Controllers & Compensators
    # -----------------------------------------------------
    elif module.startswith('Module 6'):
        st.subheader('Module 6 · Classical Controllers & Phase Compensators')
        
        comp_type = st.radio('Select Compensator Strategy', ['Lead Compensator', 'Lag Compensator', 'Lead-Lag Compensator', 'Industrial Filtered PID'])
        
        ccol1, ccol2 = st.columns([1, 2])
        with ccol1:
            if comp_type == 'Lead Compensator':
                st.markdown('**Lead Compensator Design ($0 < \\alpha < 1$)**')
                st.latex(r'G_c(s) = K \frac{T s + 1}{\alpha T s + 1}')
                k_lead = st.slider('Compensator Gain $K$', 0.1, 5.0, 1.0, 0.1)
                t_lead = st.slider('Lead Time Constant $T$', 0.1, 5.0, 1.0, 0.1)
                alpha_lead = st.slider('Attenuation Factor $\\alpha$', 0.05, 0.95, 0.2, 0.05)
                n_c, d_c = lead(k=k_lead, T=t_lead, alpha=alpha_lead)
                max_phase = np.rad2deg(np.arcsin((1.0 - alpha_lead) / (1.0 + alpha_lead)))
                w_m = 1.0 / (t_lead * np.sqrt(alpha_lead))
                st.info(f"**Max Phase Lead:** $\phi_m = {max_phase:.1f}^\circ$ at $\omega_m = {w_m:.2f}$ rad/s")

            elif comp_type == 'Lag Compensator':
                st.markdown('**Lag Compensator Design ($\\beta > 1$)**')
                st.latex(r'G_c(s) = K \frac{T s + 1}{\beta T s + 1}')
                k_lag = st.slider('Compensator Gain $K$', 0.1, 5.0, 1.0, 0.1)
                t_lag = st.slider('Lag Time Constant $T$', 0.1, 10.0, 2.0, 0.5)
                beta_lag = st.slider('Gain Boost Factor $\\beta$', 1.5, 20.0, 5.0, 0.5)
                n_c, d_c = lag(k=k_lag, T=t_lag, beta=beta_lag)
                st.info(f"**Low-Frequency Gain Boost:** $20\log_{{10}}(\\beta) = {20*np.log10(beta_lag):.1f}$ dB")

            elif comp_type == 'Lead-Lag Compensator':
                st.markdown('**Lead-Lag Compensator Design**')
                st.latex(r'G_c(s) = K \frac{(T_1 s + 1)(T_2 s + 1)}{(\alpha T_1 s + 1)(\beta T_2 s + 1)}')
                k_ll = st.slider('Compensator Gain $K$', 0.1, 5.0, 1.0, 0.1)
                t1_ll = st.slider('Lead $T_1$', 0.1, 5.0, 1.0, 0.1)
                alpha_ll = st.slider('Lead $\\alpha$', 0.05, 0.95, 0.2, 0.05)
                t2_ll = st.slider('Lag $T_2$', 0.5, 10.0, 2.0, 0.5)
                beta_ll = st.slider('Lag $\\beta$', 1.5, 20.0, 5.0, 0.5)
                n_c, d_c = lead_lag(k=k_ll, T1=t1_ll, alpha=alpha_ll, T2=t2_ll, beta=beta_ll)

            else:
                st.markdown('**Filtered Industrial PID Controller**')
                st.latex(r'C(s) = K_p + \frac{K_i}{s} + \frac{K_d s}{T_f s + 1} = \frac{(K_d + K_p T_f)s^2 + (K_p + K_i T_f)s + K_i}{T_f s^2 + s}')
                kp = st.slider('Proportional $K_p$', 0.1, 5.0, 1.5, 0.1)
                ki = st.slider('Integral $K_i$', 0.0, 3.0, 0.5, 0.1)
                kd = st.slider('Derivative $K_d$', 0.0, 2.0, 0.2, 0.05)
                tf = st.slider('Filter Time Constant $T_f$', 0.005, 0.1, 0.02, 0.005)
                n_c, d_c = controller('PID filtered', kp=kp, ki=ki, kd=kd, tf=tf)

        with ccol2:
            # Closed loop with baseline plant G_p(s) = 1 / (s + 1)
            t_comp = np.linspace(0, 15, 600)
            # Uncompensated: G(s) / (1 + G(s))
            _, y_uncomp = response([1.0], [1.0, 2.0], t_comp)
            
            # Compensated open loop: G_comp * G_p
            n_loop = np.polymul(n_c, [1.0])
            d_loop = np.polymul(d_c, [1.0, 1.0])
            # Closed loop: T = n_loop / (d_loop + n_loop)
            d_cl = np.polyadd(d_loop, np.pad(n_loop, (len(d_loop) - len(n_loop), 0)))
            _, y_comp = response(n_loop, d_cl, t_comp)

            fig_comp = make_dark_figure("Closed-Loop Step Response Comparison", height=380, xlabel="Time (s)", ylabel="Output y(t)")
            fig_comp.add_trace(go.Scatter(x=t_comp, y=y_uncomp, mode='lines', name='Uncompensated Plant', line=dict(color='#94a3b8', dash='dash')))
            fig_comp.add_trace(go.Scatter(x=t_comp, y=y_comp, mode='lines', name=f'Compensated with {comp_type}', line=dict(color='#00f2fe', width=2.5)))
            fig_comp.add_hline(y=1.0, line_dash='dot', line_color='#64748b', annotation_text='Target = 1.0')
            st.plotly_chart(fig_comp, use_container_width=True)

    # -----------------------------------------------------
    # MODULE 7: State-Space & Modern Control Analysis
    # -----------------------------------------------------
    else:
        st.subheader('Module 7 · State-Space & Modern Control Analysis')
        st.markdown('''
        Continuous-time linear time-invariant (LTI) state representation of the four thermal zones:
        ''')
        st.latex(r'\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t) + \mathbf{E}\mathbf{q}(t), \quad \mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)')
        
        A_ss, B_ss, C_ss, D_ss, E_ss = linearize()
        
        ss_tabs = st.tabs(['Matrices & Eigenstructure', 'Controllability & Observability', 'State Transition Matrix $\Phi(t)$', 'Free Dynamic Simulation'])
        
        with ss_tabs[0]:
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown('**State Matrix $\mathbf{A}$:**')
                st.latex(r'\mathbf{A} = ' + matrix_to_latex(A_ss))
                st.markdown('**Actuator Matrix $\mathbf{B}$:**')
                st.latex(r'\mathbf{B} = ' + matrix_to_latex(B_ss))
            with sc2:
                st.markdown('**Disturbance Matrix $\mathbf{E}$:**')
                st.latex(r'\mathbf{E} = ' + matrix_to_latex(E_ss))
                eigvals_ss, eigvecs_ss = np.linalg.eig(A_ss)
                st.markdown('**Eigenvalues & Time Constants:**')
                for idx, ev in enumerate(eigvals_ss):
                    tau_mode = -1.0 / ev if ev < 0 else np.nan
                    st.latex(fr"\lambda_{{{idx+1}}} = {ev:.5f}\text{{ s}}^{{-1}} \implies \tau_{{{idx+1}}} = {tau_mode:.1f}\text{{ s}}")

        with ss_tabs[1]:
            st.markdown('#### Controllability & Observability Tests')
            # Controllability
            C_mat = ctrb(A_ss, B_ss)
            rank_ctrb = np.linalg.matrix_rank(C_mat)
            # Observability
            O_mat = obsv(A_ss, C_ss)
            rank_obsv = np.linalg.matrix_rank(O_mat)
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                card_ctrb_color = '#10b981' if rank_ctrb == 4 else '#f43f5e'
                card_ctrb_msg = 'Full state controllability established. Any state can be steered within finite time.' if rank_ctrb == 4 else 'System is not fully controllable.'
                c_col1.markdown(f'<div class="tg-card" style="border-left:4px solid {card_ctrb_color};"><h4 style="color:#ffffff; margin:0;">Controllability Matrix $\\mathcal{{C}}$</h4><div style="font-size:18px; font-weight:700; color:#38bdf8; margin:8px 0;">Rank: {rank_ctrb} / 4</div><div style="font-size:12px; color:#94a3b8;">{card_ctrb_msg}</div></div>', unsafe_allow_html=True)
                c_col1.caption(f"Shape of $\\mathcal{{C}} = [B \\quad AB \\quad A^2B \\quad A^3B]$: {C_mat.shape}")

            with c_col2:
                card_obsv_color = '#10b981' if rank_obsv == 4 else '#f43f5e'
                card_obsv_msg = 'Full observability established. Initial state can be reconstructed uniquely from measured outputs.' if rank_obsv == 4 else 'System is not fully observable.'
                c_col2.markdown(f'<div class="tg-card" style="border-left:4px solid {card_obsv_color};"><h4 style="color:#ffffff; margin:0;">Observability Matrix $\\mathcal{{O}}$</h4><div style="font-size:18px; font-weight:700; color:#34d399; margin:8px 0;">Rank: {rank_obsv} / 4</div><div style="font-size:12px; color:#94a3b8;">{card_obsv_msg}</div></div>', unsafe_allow_html=True)
                c_col2.caption(f"Shape of $\\mathcal{{O}} = [C^T \\quad A^TC^T \\quad (A^T)^2C^T \\quad (A^T)^3C^T]^T$: {O_mat.shape}")

        with ss_tabs[2]:
            st.markdown('#### State Transition Matrix $\Phi(t) = e^{\mathbf{A}t}$')
            t_eval = st.slider('Evaluation Time $t$ (s)', 1.0, 100.0, 10.0, 1.0)
            phi_ch = cayley_hamilton_expm(A_ss, t_eval)
            phi_ref = expm(A_ss * t_eval)
            diff_norm = np.linalg.norm(phi_ch - phi_ref)
            
            st.latex(r'\Phi(' + str(int(t_eval)) + r'\text{ s}) = ' + matrix_to_latex(phi_ch))
            st.markdown(f'''
            <div style="font-size:13px; color:#94a3b8; margin-top:8px;">
                • <strong>Method:</strong> Cayley-Hamilton polynomial interpolation at eigenvalues.<br/>
                • <strong>Validation vs scipy.linalg.expm:</strong> Frobenius difference $\|\Phi_{{CH}} - e^{{At}}\|_F = {diff_norm:.2e}$
            </div>
            ''', unsafe_allow_html=True)

        with ss_tabs[3]:
            st.markdown('#### Unforced Free Transient Response from Initial Offset')
            st.caption('Evolution of zone temperatures in the absence of actuation or active fault, demonstrating thermal dissipation and inter-zone diffusion.')
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            t1_init = f_col1.slider('Zone 1 $\Delta T(0)$ (°C)', 0.0, 20.0, 15.0)
            t2_init = f_col2.slider('Zone 2 $\Delta T(0)$ (°C)', 0.0, 20.0, 5.0)
            t3_init = f_col3.slider('Zone 3 $\Delta T(0)$ (°C)', 0.0, 20.0, 2.0)
            t4_init = f_col4.slider('Zone 4 $\Delta T(0)$ (°C)', 0.0, 20.0, 0.0)
            
            x0 = np.array([t1_init, t2_init, t3_init, t4_init])
            t_free = np.linspace(0, 1200, 300)
            x_free = np.array([expm(A_ss * tk) @ x0 for tk in t_free])
            
            fig_free = make_dark_figure("Autonomous Zone Thermal Dissipation", height=380, xlabel="Time (s)", ylabel="Temperature Deviation ΔT (°C)")
            for i in range(4):
                fig_free.add_trace(go.Scatter(x=t_free, y=x_free[:, i], mode='lines', name=f'Zone {i+1}', line=dict(color=ZONE_COLORS[i], width=2.2)))
            st.plotly_chart(fig_free, use_container_width=True)

# =========================================================
# WORKSPACE 3: PROJECT INFORMATION & SYLLABUS GUIDE
# =========================================================
else:
    st.markdown('''
    <div style="margin-bottom: 20px;">
        <h1 style="font-size:28px; font-weight:800; color:#f8fafc; margin-bottom:4px;">Project Documentation & Syllabus Alignment</h1>
        <p style="color:#94a3b8; font-size:14px; margin:0;">
            Comprehensive academic guide to ThermoGuard's architecture, thermal equations, and control theory mappings.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="tg-card">
        <h3 style="color:#f8fafc; margin-top:0;">⚡ Executive Summary</h3>
        <p style="color:#cbd5e1; line-height:1.7;">
            <strong>ThermoGuard</strong> is an interactive educational and laboratory platform simulating four thermally coupled zones in a grid-scale 
            Battery Energy Storage System (BESS) enclosure. It connects non-linear thermal dynamics with classical SISO and modern state-space control engineering, 
            providing transparent benchmarking between <em>Open-Loop Fixed Cooling</em>, <em>Conventional Average-Temperature PI/PID Feedback</em>, and a 
            <em>Zone-Aware Priority Allocation Heuristic</em>.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('### Control Systems Syllabus Matrix')
    df_syllabus = pd.DataFrame([
        {'Module': 'Module 1', 'Curriculum Topic': 'Open/Closed Loop, Negative Feedback, Block Diagram Reduction', 'Mathematical Tool': '`block_diagrams.py`, `time_domain.py`', 'Physical BESS Context': 'Closed-loop cooling actuation, sensor feedback'},
        {'Module': 'Module 2', 'Curriculum Topic': 'Differential Equations, Thermal RC Analogy, Mason Gain Formula', 'Mathematical Tool': '`thermal_model.py`, `signal_flow.py`, `electrical_models.py`', 'Physical BESS Context': 'Coupled 4-zone lumped RC circuit, RLC/mass-spring analogy'},
        {'Module': 'Module 3', 'Curriculum Topic': 'Step/Ramp/Impulse, Transient Time Specs (Rise, Settling, Peak)', 'Mathematical Tool': '`time_domain.py`, SciPy Lsim/Step', 'Physical BESS Context': 'Dynamic response to battery discharge steps and heat shocks'},
        {'Module': 'Module 4', 'Curriculum Topic': 'Poles/Zeros, Characteristic Equation, Routh-Hurwitz, Root Locus', 'Mathematical Tool': '`stability.py`, Polynomial Roots', 'Physical BESS Context': 'Stability boundaries under varied convection and cooling gains'},
        {'Module': 'Module 5', 'Curriculum Topic': 'Bode Diagrams, Polar/Nyquist Locus, Gain & Phase Margins', 'Mathematical Tool': '`frequency_domain.py`, SciPy Freqs', 'Physical BESS Context': 'Robustness against thermal sensor delay and crossover behavior'},
        {'Module': 'Module 6', 'Curriculum Topic': 'P, PI, Filtered PID, Lead/Lag/Lead-Lag Phase Compensation', 'Mathematical Tool': '`compensators.py`, TransferFunction', 'Physical BESS Context': 'Actuator lead phase boost and steady-state offset elimination'},
        {'Module': 'Module 7', 'Curriculum Topic': 'State-Space Realization, Controllability, Observability, Transition', 'Mathematical Tool': '`state_space.py`, Cayley-Hamilton', 'Physical BESS Context': 'MIMO coupled temperature states, sensor reconstructibility'}
    ])
    st.dataframe(df_syllabus, use_container_width=True, hide_index=True)

    st.markdown('### Thermal Modelling & Allocation Methodology')
    st.markdown('''
    #### Zone-Aware Priority Heuristic
    ThermoGuard formulates a real-time priority score $P_i$ for each zone $i$:
    ''')
    st.latex(r'P_i = 0.15 + \max(T_{i,\text{meas}} - T_{\text{setpoint}}, 0) + 20 \max\left(\frac{dT_{i,\text{meas}}}{dt}, 0\right) + \frac{\max(Q_i, 0)}{5000}')
    st.markdown('''
    Desired cooling commands are computed incorporating heat feedforward and actuator health compensation:
    ''')
    st.latex(r'u_{i,\text{desired}} = \frac{\text{clip}\left(0.18 + 0.16 \max(T_i - T_{\text{sp}}, 0) + \frac{Q_i}{90000}, 0, 1\right)}{\max(\eta_i, 0.15)}')
    st.markdown('''
    The total cooling budget $\sum u_i \le U_{\max}$ is allocated proportionally to priority $w_i = P_i / \sum P_j$, followed by iterative redistribution of unused capacity to zones possessing actuator headroom.
    ''')

    st.markdown('''
    <div class="tg-card" style="border-left: 4px solid #fbbf24; margin-top:20px;">
        <h4 style="color:#fbbf24; margin-top:0;">Educational Scope & Model Limitations</h4>
        <div style="font-size:13px; color:#cbd5e1; line-height:1.6;">
            All thermal capacitances, convection coefficients, and heat coefficients are illustrative parameters intended for educational and control-systems analysis. 
            The model omits internal electrochemical kinetics, cell-level spatial gradients, and complex refrigeration cycles. 
            The fan energy metric $\int \sum u_i^3 dt$ is a cubic fan-affinity proxy rather than a calibrated HVAC compressor energy model.
        </div>
    </div>
    ''', unsafe_allow_html=True)
