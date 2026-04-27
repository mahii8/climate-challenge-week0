import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\bamla\OneDrive\Desktop\climate-challenge-week0')

def run_eda(country_name):
    print(f"\n{'='*50}")
    print(f"  Running EDA for {country_name}")
    print(f"{'='*50}")

    # Load data
    filename = f"data/{country_name.lower()}.csv"
    df = pd.read_csv(filename)
    df['Country'] = country_name

    # Date parsing
    df['Date'] = pd.to_datetime(df['YEAR'] * 1000 + df['DOY'], format='%Y%j')
    df['Month'] = df['Date'].dt.month

    # Replace -999 with NaN
    df.replace(-999, np.nan, inplace=True)

    # Drop duplicates
    dupes = df.duplicated().sum()
    print(f"Duplicates found: {dupes}")
    df.drop_duplicates(inplace=True)

    # Missing value report
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    print("\nMissing Values (%):")
    print(missing_pct[missing_pct > 0])

    # Outlier detection
    cols = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX']
    z_scores = np.abs(stats.zscore(df[cols].dropna()))
    print("\nOutliers (|Z| > 3):")
    for i, col in enumerate(cols):
        print(f"  {col}: {(z_scores[:, i] > 3).sum()}")

    # Forward fill
    weather_cols = ['T2M', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE',
                    'PRECTOTCORR', 'RH2M', 'WS2M', 'WS2M_MAX', 'PS', 'QV2M']
    df[weather_cols] = df[weather_cols].ffill()

    # Drop rows with >30% missing
    threshold = len(df.columns) * 0.3
    df.dropna(thresh=int(threshold), inplace=True)

    # Export cleaned data
    df.to_csv(f"data/{country_name.lower()}_clean.csv", index=False)
    print(f"\n✅ Cleaned data saved: data/{country_name.lower()}_clean.csv")
    print(f"   Shape: {df.shape}")

    # ── PLOTS ──────────────────────────────────────────
    plt.rcParams['figure.figsize'] = (12, 5)
    sns.set_theme(style='whitegrid')

    # 1. Temperature time series
    monthly_temp = df.groupby(df['Date'].dt.to_period('M'))['T2M'].mean()
    monthly_temp.index = monthly_temp.index.to_timestamp()
    warmest = monthly_temp.idxmax()
    coolest = monthly_temp.idxmin()

    plt.figure()
    plt.plot(monthly_temp.index, monthly_temp.values, color='tomato', linewidth=1.5)
    plt.annotate(f'Warmest\n{warmest.strftime("%b %Y")}',
                 xy=(warmest, monthly_temp[warmest]),
                 xytext=(30, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), color='red')
    plt.annotate(f'Coolest\n{coolest.strftime("%b %Y")}',
                 xy=(coolest, monthly_temp[coolest]),
                 xytext=(30, -20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'), color='blue')
    plt.title(f'{country_name} — Monthly Average Temperature (2015–2026)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig(f'notebooks/{country_name.lower()}_temp_timeseries.png', dpi=150)
    plt.show()

    # 2. Precipitation bar chart
    monthly_precip = df.groupby(df['Date'].dt.month)['PRECTOTCORR'].sum()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    peak_month = monthly_precip.idxmax() - 1

    plt.figure()
    bars = plt.bar(month_names, monthly_precip.values, color='steelblue')
    bars[peak_month].set_color('darkblue')
    plt.title(f'{country_name} — Monthly Total Precipitation (2015–2026)')
    plt.xlabel('Month')
    plt.ylabel('Total Precipitation (mm)')
    plt.tight_layout()
    plt.savefig(f'notebooks/{country_name.lower()}_precip_bar.png', dpi=150)
    plt.show()
    print(f"   Peak rainy month: {month_names[peak_month]}")

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[weather_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f'{country_name} — Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'notebooks/{country_name.lower()}_correlation_heatmap.png', dpi=150)
    plt.show()

    # 4. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df['T2M'], df['RH2M'], alpha=0.3, color='teal', s=10)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Relative Humidity (%)')
    axes[0].set_title('T2M vs RH2M')
    axes[1].scatter(df['T2M_RANGE'], df['WS2M'], alpha=0.3, color='coral', s=10)
    axes[1].set_xlabel('Temperature Range (°C)')
    axes[1].set_ylabel('Wind Speed (m/s)')
    axes[1].set_title('T2M_RANGE vs WS2M')
    plt.suptitle(f'{country_name} — Scatter Plots')
    plt.tight_layout()
    plt.savefig(f'notebooks/{country_name.lower()}_scatter_plots.png', dpi=150)
    plt.show()

    # 5. Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['PRECTOTCORR'].dropna(), bins=50, color='steelblue', edgecolor='white')
    axes[0].set_yscale('log')
    axes[0].set_title('Distribution of Daily Precipitation (log scale)')
    axes[0].set_xlabel('Precipitation (mm/day)')
    axes[0].set_ylabel('Frequency (log)')
    sample = df.sample(min(500, len(df)), random_state=42)
    axes[1].scatter(sample['T2M'], sample['RH2M'],
                    s=sample['PRECTOTCORR'].fillna(0) * 10 + 5,
                    alpha=0.4, color='purple')
    axes[1].set_title('Bubble Chart: T2M vs RH2M (size = Precipitation)')
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('Relative Humidity (%)')
    plt.suptitle(f'{country_name} — Distributions')
    plt.tight_layout()
    plt.savefig(f'notebooks/{country_name.lower()}_distributions.png', dpi=150)
    plt.show()

    print(f"\n✅ {country_name} EDA complete! All plots saved.")
    return df

