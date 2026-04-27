import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="African Climate Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ── Load data ───────────────────────────────────────────
@st.cache_data
def load_data():
    countries = ['ethiopia', 'kenya', 'sudan', 'tanzania', 'nigeria']
    dfs = []
    for country in countries:
        filepath = os.path.join('data', f'{country}_clean.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Country'] = country.capitalize()
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_data()

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Flag_of_Ethiopia.svg/320px-Flag_of_Ethiopia.svg.png", width=100)
st.sidebar.title("🌍 Dashboard Controls")

# Country selector
all_countries = sorted(df['Country'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=all_countries
)

# Year range slider
min_year = int(df['Date'].dt.year.min())
max_year = int(df['Date'].dt.year.max())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Variable selector
variable_options = {
    'Mean Temperature (T2M)': 'T2M',
    'Max Temperature (T2M_MAX)': 'T2M_MAX',
    'Min Temperature (T2M_MIN)': 'T2M_MIN',
    'Precipitation (PRECTOTCORR)': 'PRECTOTCORR',
    'Relative Humidity (RH2M)': 'RH2M',
    'Wind Speed (WS2M)': 'WS2M'
}
selected_variable_label = st.sidebar.selectbox(
    "Select Climate Variable",
    options=list(variable_options.keys())
)
selected_variable = variable_options[selected_variable_label]

# ── Filter data ─────────────────────────────────────────
filtered = df[
    (df['Country'].isin(selected_countries)) &
    (df['Date'].dt.year >= year_range[0]) &
    (df['Date'].dt.year <= year_range[1])
]

# ── Main content ────────────────────────────────────────
st.title("🌍 African Climate Trend Dashboard")
st.markdown("**EthioClimate Analytics** | Supporting Ethiopia's COP32 Preparations | Data: NASA POWER (2015–2026)")
st.markdown("---")

# ── KPI Cards ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_temp = filtered['T2M'].mean()
    st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f} °C")

with col2:
    avg_precip = filtered['PRECTOTCORR'].mean()
    st.metric("🌧️ Avg Precipitation", f"{avg_precip:.2f} mm/day")

with col3:
    heat_days = (filtered['T2M_MAX'] > 35).sum()
    st.metric("🔥 Extreme Heat Days", f"{heat_days:,}")

with col4:
    dry_days = (filtered['PRECTOTCORR'] < 1).sum()
    st.metric("🏜️ Dry Days", f"{dry_days:,}")

st.markdown("---")

# ── Row 1: Temperature trend + Precipitation boxplot ───
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📈 {selected_variable_label} Trend Over Time")
    monthly = filtered.groupby(
        ['Country', filtered['Date'].dt.to_period('M')]
    )[selected_variable].mean().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['tomato', 'steelblue', 'green', 'orange', 'purple']
    for i, country in enumerate(selected_countries):
        data = monthly[monthly['Country'] == country]
        ax.plot(data['Date'], data[selected_variable],
                label=country, linewidth=1.5,
                color=colors[i % len(colors)])
    ax.set_xlabel('Date')
    ax.set_ylabel(selected_variable_label)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Monthly Average {selected_variable_label}')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📦 Precipitation Distribution by Country")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=filtered, x='Country', y='PRECTOTCORR',
                palette=colors[:len(selected_countries)],
                showfliers=False, ax=ax)
    ax.set_xlabel('Country')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_title('Daily Precipitation Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Row 2: Extreme heat + Dry days ─────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Extreme Heat Days per Year (T2M_MAX > 35°C)")
    heat = filtered[filtered['T2M_MAX'] > 35].groupby(
        ['Country', filtered['Date'].dt.year]
    ).size().reset_index(name='Heat_Days')
    heat.columns = ['Country', 'Year', 'Heat_Days']
    heat_avg = heat.groupby('Country')['Heat_Days'].mean().round(1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(heat_avg.index, heat_avg.values,
                  color=colors[:len(heat_avg)], edgecolor='white')
    for bar, val in zip(bars, heat_avg.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val}', ha='center', fontsize=10)
    ax.set_ylabel('Avg Days per Year')
    ax.set_title('Average Extreme Heat Days')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("📊 Temperature Summary Table")
    summary = filtered.groupby('Country')['T2M'].agg(
        Mean='mean',
        Median='median',
        Std='std',
        Max='max',
        Min='min'
    ).round(2)
    st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ── Footer ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
Built with Streamlit | Data: NASA POWER | 
EthioClimate Analytics — 10 Academy x Kifiya Week 0 Challenge
</div>
""", unsafe_allow_html=True)