import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

print("=" * 62)
print("    🌍 GLOBAL HEALTH SNAPSHOT — FULL DATA PIPELINE")
print("=" * 62)

# ── STEP 1 — GENERATE DATASET ────────────────────────────────
# 10 countries over 20 years (2000–2019)
np.random.seed(42)

countries = [
    "UK", "USA", "India", "Brazil", "Germany", 
    "Japan", "China", "Kenya", "Canada", "Australia"
]
years = list(range(2000, 2020))

# Base configuration dictionary for initial metrics
base_data = {
    "UK": {"le": 78, "gdp": 35000, "hs": 9.5, "im": 5.2, "vac": 94},
    "USA": {"le": 77, "gdp": 45000, "hs": 15.0, "im": 6.1, "vac": 91},
    "India": {"le": 62, "gdp": 1200, "hs": 3.5, "im": 45.0, "vac": 75},
    "Brazil": {"le": 70, "gdp": 6000, "hs": 8.0, "im": 22.0, "vac": 88},
    "Germany": {"le": 79, "gdp": 38000, "hs": 10.5, "im": 4.1, "vac": 93},
    "Japan": {"le": 81, "gdp": 39000, "hs": 9.8, "im": 2.8, "vac": 95},
    "China": {"le": 72, "gdp": 3000, "hs": 4.8, "im": 30.0, "vac": 85},
    "Kenya": {"le": 54, "gdp": 900, "hs": 4.5, "im": 55.0, "vac": 70},
    "Canada": {"le": 79, "gdp": 40000, "hs": 10.0, "im": 5.0, "vac": 92},
    "Australia": {"le": 80, "gdp": 42000, "hs": 9.2, "im": 4.5, "vac": 93}
}

records = []
for country in countries:
    b = base_data[country]
    for i, year in enumerate(years):
        records.append({
            "Country": country,
            "Year": year,
            "Life_Expectancy": round(b["le"] + 0.15 * i + np.random.normal(0, 0.4), 2),
            "GDP_Per_Capita": round(b["gdp"] + 800 * i + np.random.normal(0, 200), 2),
            "Healthcare_Spend": round(b["hs"] + 0.05 * i + np.random.normal(0, 0.2), 2),
            "Infant_Mortality": round(max(1.0, b["im"] - 1.2 * i + np.random.normal(0, 0.5)), 2),
            "Vaccinations": round(min(99.0, b["vac"] + 0.3 * i + np.random.normal(0, 0.5)), 2)
        })

df = pd.DataFrame(records)
print(f"\n📂 STEP 1: Dataset created successfully ({len(df)} rows, {df.shape[1]} columns)")

# ── STEP 2 — INJECT & DETECT MISSING VALUES ────────────────────
numeric_cols = [
    "Life_Expectancy", "GDP_Per_Capita", 
    "Healthcare_Spend", "Infant_Mortality", "Vaccinations"
]

# Randomly inject 12 missing values
for _ in range(12):
    idx = np.random.choice(df.index)
    col = np.random.choice(numeric_cols)
    df.loc[idx, col] = np.nan

missing_count_before = df[numeric_cols].isnull().sum().sum()
print(f"\n🧹 STEP 2: Missing values injected: {missing_count_before}")

# ── STEP 3 — CLEAN WITH fillna() ──────────────────────────────
for col in numeric_cols:
    col_mean = df[col].mean()
    df[col] = df[col].fillna(col_mean)

missing_count_after = df[numeric_cols].isnull().sum().sum()
print(f"   Missing values after fillna(): {missing_count_after} (All clean! ✅)")

# ── STEP 4 — STATISTICAL ANALYSIS ──────────────────────────────
print("\n📊 STEP 4 — Statistical Summary & Insights")
print("-" * 50)
print(df[numeric_cols].describe())

top_country = df.groupby("Country")["Life_Expectancy"].mean().idxmax()
print(f"\n✨ Country with highest average Life Expectancy: {top_country}")

correlation = df[["GDP_Per_Capita", "Life_Expectancy"]].corr().loc["GDP_Per_Capita", "Life_Expectancy"]
print(f"🔗 Correlation between GDP per Capita and Life Expectancy: {correlation:.2f}")

# ── STEP 5 — TREND LINE WITH polyfit() ─────────────────────────
# Extract UK data as an example for polynomial regression
uk_df = df[df["Country"] == "UK"].copy()
x = uk_df["Year"].values           # Convert to NumPy array via .values
y = uk_df["Life_Expectancy"].values

coeffs = np.polyfit(x, y, deg=1)     # Fit degree-1 polynomial (slope & intercept)
trend_fn = np.poly1d(coeffs)       # Create callable trend function
y_trend = trend_fn(x)              # Generate trend values

# ── STEP 6 — MULTI-PANEL VISUALISATION ─────────────────────────
print("\n🎨 STEP 6 — Generating Multi-Panel Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Global Health Snapshot Dashboard (2000–2019)", fontsize=16, fontweight="bold")
fig.patch.set_facecolor("#f8f9fa")

# Panel 1 — Line Chart: Life Expectancy Trends for all countries
ax1 = axes[0, 0]
for country in countries:
    cdf = df[df["Country"] == country]
    ax1.plot(cdf["Year"], cdf["Life_Expectancy"], alpha=0.7, label=country)
ax1.set_title("Life Expectancy Trends (All Countries)", fontweight="bold")
ax1.set_xlabel("Year")
ax1.set_ylabel("Life Expectancy (Years)")
ax1.grid(alpha=0.3)
ax1.set_facecolor("white")

# Panel 2 — Scatter + Trend Line: GDP vs Life Expectancy
ax2 = axes[0, 1]
ax2.scatter(df["GDP_Per_Capita"], df["Life_Expectancy"], color="teal", alpha=0.6, edgecolors="white", label="Data Points")
# Fit global trend for scatter
gdp_x = df["GDP_Per_Capita"].values
le_y = df["Life_Expectancy"].values
sc_coeffs = np.polyfit(gdp_x, le_y, 1)
sc_fn = np.poly1d(sc_coeffs)
sorted_idx = np.argsort(gdp_x)
ax2.plot(gdp_x[sorted_idx], sc_fn(gdp_x[sorted_idx]), color="darkred", linestyle="--", linewidth=2, label="Trend Line")
ax2.set_title("GDP per Capita vs Life Expectancy", fontweight="bold")
ax2.set_xlabel("GDP per Capita (USD)")
ax2.set_ylabel("Life Expectancy (Years)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_facecolor("white")

# Panel 3 — Horizontal Bar: Average Vaccination Rate per Country (Sorted)
ax3 = axes[1, 0]
avg_vac = df.groupby("Country")["Vaccinations"].mean().sort_values()
ax3.barh(avg_vac.index, avg_vac.values, color="cornflowerblue", edgecolor="white")
ax3.set_title("Average Vaccination Coverage by Country", fontweight="bold")
ax3.set_xlabel("Vaccination Coverage (%)")
ax3.set_xlim(60, 100)
ax3.grid(axis="x", alpha=0.3)
ax3.set_facecolor("white")

# Panel 4 — Line Chart: Healthcare Spend + axhline() Reference Line
ax4 = axes[1, 1]
global_mean_spend = df["Healthcare_Spend"].mean()
for country in ["USA", "Germany", "India", "Kenya"]:
    cdf = df[df["Country"] == country]
    ax4.plot(cdf["Year"], cdf["Healthcare_Spend"], marker="o", markersize=3, label=country)

# Add reference line using axhline()
ax4.axhline(global_mean_spend, color="red", linestyle="--", linewidth=1.5, 
            label=f"Global Mean ({global_mean_spend:.1f}%)")
ax4.set_title("Healthcare Spending Over Time & Global Mean", fontweight="bold")
ax4.set_xlabel("Year")
ax4.set_ylabel("Healthcare Spend (% of GDP)")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)
ax4.set_facecolor("white")

# Format and Save Figure
plt.tight_layout()
plt.savefig("health_snapshot.png", dpi=150, bbox_inches="tight")
print("  ✅ Saved figure as: health_snapshot.png")
plt.show()

# ── STEP 7 — EXPORT PIPELINE SUMMARY ───────────────────────────
df.to_csv("health_full.csv", index=False)
print("💾 STEP 8 — Pipeline Complete: Exported dataset to 'health_full.csv'.")
print("=" * 62)